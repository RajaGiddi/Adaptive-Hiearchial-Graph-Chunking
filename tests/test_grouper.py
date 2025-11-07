import json

import pytest

import ahgc.llm.grouper as g


@pytest.fixture
def sample_segments():
    return [
        {"segment_id": "seg_0", "summary": "Intro"},
        {"segment_id": "seg_1", "summary": "Details"},
        {"segment_id": "seg_2", "summary": "Conclusion"},
    ]


def _noop_agent():
    # simple sentinel object the code won't use directly
    return object()


def test_induce_hierarchy_valid_json(monkeypatch, sample_segments):
    # Avoid env/API by stubbing both calls
    monkeypatch.setattr(g, "get_agent", lambda **kwargs: _noop_agent())
    monkeypatch.setattr(
        g,
        "generate_with_agent",
        lambda agent, prompt: json.dumps(
            {
                "sections": [
                    {
                        "section_id": "s1",
                        "section_title": "All",
                        "segment_ids": ["seg_0", "seg_1", "seg_2"],
                    }
                ]
            }
        ),
    )

    out = g.induce_hierarchy(sample_segments)
    assert list(out.keys()) == ["sections"]
    assert out["sections"][0]["segment_ids"] == ["seg_0", "seg_1", "seg_2"]


def test_induce_hierarchy_parse_failure_fallback(monkeypatch, sample_segments):
    monkeypatch.setattr(g, "get_agent", lambda **kwargs: _noop_agent())
    monkeypatch.setattr(g, "generate_with_agent", lambda agent, prompt: "not json")

    out = g.induce_hierarchy(sample_segments)
    assert out["sections"][0]["section_title"] == "All Segments"
    assert out["sections"][0]["segment_ids"] == ["seg_0", "seg_1", "seg_2"]


def test_induce_hierarchy_unknown_id_fallback(monkeypatch, sample_segments):
    monkeypatch.setattr(g, "get_agent", lambda **kwargs: _noop_agent())
    monkeypatch.setattr(
        g,
        "generate_with_agent",
        lambda agent, prompt: json.dumps(
            {
                "sections": [
                    {
                        "section_id": "s1",
                        "section_title": "Weird",
                        "segment_ids": ["seg_0", "seg_999", "seg_1"],
                    }
                ]
            }
        ),
    )

    out = g.induce_hierarchy(sample_segments)
    assert out["sections"][0]["segment_ids"] == ["seg_0", "seg_1", "seg_2"]


def test_induce_hierarchy_out_of_order_fallback(monkeypatch, sample_segments):
    monkeypatch.setattr(g, "get_agent", lambda **kwargs: _noop_agent())
    monkeypatch.setattr(
        g,
        "generate_with_agent",
        lambda agent, prompt: json.dumps(
            {
                "sections": [
                    {
                        "section_id": "s1",
                        "section_title": "Wrong order",
                        "segment_ids": ["seg_1", "seg_0", "seg_2"],
                    }
                ]
            }
        ),
    )

    out = g.induce_hierarchy(sample_segments)
    assert out["sections"][0]["segment_ids"] == ["seg_0", "seg_1", "seg_2"]
