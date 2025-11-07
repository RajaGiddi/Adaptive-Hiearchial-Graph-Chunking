from ahgc.chunking.materialize import materialize_sections


def test_materialize_single_section():
    segments = [
        {"segment_id": "seg_0", "text": "A"},
        {"segment_id": "seg_1", "text": "B"},
    ]
    hierarchy = {
        "sections": [
            {
                "section_id": "s1",
                "section_title": "All",
                "segment_ids": ["seg_0", "seg_1"],
            }
        ]
    }

    sections = materialize_sections(hierarchy, segments)
    assert len(sections) == 1
    s = sections[0]
    assert s["section_id"] == "s1"
    assert s["section_title"] == "All"
    assert s["text"] == "A\n\nB"


def test_materialize_multiple_sections_and_missing():
    segments = [
        {"segment_id": "seg_0", "text": "A"},
        {"segment_id": "seg_1", "text": "B"},
        # seg_2 missing intentionally
        {"segment_id": "seg_3", "text": "D"},
    ]
    hierarchy = {
        "sections": [
            {
                "section_id": "s1",
                "section_title": "First",
                "segment_ids": ["seg_0", "seg_2"],
            },
            {
                "section_id": "s2",
                "section_title": "Second",
                "segment_ids": ["seg_1", "seg_3"],
            },
        ]
    }

    sections = materialize_sections(hierarchy, segments)
    assert len(sections) == 2
    s1, s2 = sections
    assert s1["text"] == "A"  # seg_2 skipped
    assert s2["text"] == "B\n\nD"
