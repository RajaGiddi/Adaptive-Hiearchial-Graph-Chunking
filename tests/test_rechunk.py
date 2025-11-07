from ahgc.chunking.rechunk import (
    rechunk_section,
    build_final_chunks,
    iter_rechunk_section,
    iter_final_chunks,
)


def make_section(text: str, section_id: str = "s1", title: str = "Title"):
    return {"section_id": section_id, "section_title": title, "text": text}


def test_rechunk_section_overlap_and_sizes():
    # 24 tokens, window 10, overlap 2 => starts at 0,8,16 => sizes: 10,10,8
    text = " ".join([f"w{i}" for i in range(24)])
    sec = make_section(text, section_id="s1")
    chunks = rechunk_section(sec, max_tokens=10, overlap=2)
    assert [len(c["text"].split()) for c in chunks] == [10, 10, 8]
    # Check overlap: end of chunk0 tokens should equal start of chunk1 tokens
    c0_tokens = chunks[0]["text"].split()
    c1_tokens = chunks[1]["text"].split()
    assert c0_tokens[-2:] == c1_tokens[:2]


def test_build_final_chunks_chunk_ids():
    sections = [
        make_section("one two three four five six", section_id="sX", title="T"),
    ]
    out = build_final_chunks("doc1", sections, max_tokens=3, overlap=1)
    # Expect chunk ids like doc1-sX-0, doc1-sX-1, ...
    assert all(ch["chunk_id"].startswith("doc1-sX-") for ch in out)
    assert out[0]["section_id"] == "sX"
    assert out[0]["section_title"] == "T"


def test_short_text_single_chunk_and_stream_equivalence():
    text = "Short text fits in one window"
    sec = make_section(text, section_id="s2")
    eager = rechunk_section(sec, max_tokens=50, overlap=10)
    streamed = list(iter_rechunk_section(sec, max_tokens=50, overlap=10))
    assert len(eager) == 1
    assert eager == streamed


def test_long_text_multiple_chunks_stream_equivalence():
    text = " ".join([f"w{i}" for i in range(100)])  # 100 tokens
    sec = make_section(text, section_id="s3")
    eager = rechunk_section(sec, max_tokens=25, overlap=5)
    streamed = list(iter_rechunk_section(sec, max_tokens=25, overlap=5))
    assert len(eager) > 1
    assert eager == streamed


def test_iter_final_chunks_matches_build_final_chunks():
    sections = [
        make_section(" ".join([f"w{i}" for i in range(30)]), section_id="sA"),
        make_section(" ".join([f"x{i}" for i in range(30)]), section_id="sB"),
    ]
    eager = build_final_chunks("docZ", sections, max_tokens=15, overlap=3)
    streamed = list(iter_final_chunks("docZ", sections, max_tokens=15, overlap=3))
    assert eager == streamed
