def test_ingest_demo_runs(tmp_path):
    import subprocess, json, re
    result = subprocess.run(
        ["python", "scripts/ingest_doc.py", "--input", "sample_docs/demo.md"],
        capture_output=True,
        text=True,
    )
    assert "=== AHGC Ingestion Summary" in result.stdout
    m = re.search(r"\{[\s\S]+\}", result.stdout)
    assert m, "No JSON report found"
    report = json.loads(m.group(0))
    assert report["num_segments"] > 0
    assert report["num_graph_nodes"] >= report["num_chunks"]
