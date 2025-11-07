from ahgc.llm.summarizer import summarize_segment, summarize_segments


def test_summarize_segment_short_circuit():
	text = "Short segment"  # < 40 chars
	summary = summarize_segment(text)
	assert summary == text


def test_summarize_segments_short_circuit_batch():
	segments = [
		{"segment_id": "seg_0", "text": "Alpha"},
		{"segment_id": "seg_1", "text": "Beta"},
	]
	out = summarize_segments(segments)
	assert all(len(s["text"]) < 40 for s in segments)
	assert [d["summary"] for d in out] == ["Alpha", "Beta"]

