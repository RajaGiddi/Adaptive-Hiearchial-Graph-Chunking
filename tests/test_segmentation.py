from ahgc.ingestion.segmentation import segment_document


def test_segment_document_simple():
	out = segment_document("A\n\nB")
	assert [d["segment_id"] for d in out] == ["seg_0", "seg_1"]
	assert [d["text"] for d in out] == ["A", "B"]
	assert [d["order"] for d in out] == [0, 1]


def test_segment_document_skips_empty():
	out = segment_document("A\n\n\nB")
	assert len(out) == 2
	assert out[0]["text"] == "A"
	assert out[1]["text"] == "B"

