from spoiler_gen.evaluation.metrics import compute_bleu4, compute_metrics_by_spoiler_type


def test_compute_bleu4():
    preds = ["this is a test", "hello world nice day"]
    refs = ["this is a test", "hello world nice day"]
    score = compute_bleu4(preds, refs)
    assert score > 99.0


def test_compute_metrics_by_spoiler_type():
    # Upgrade test sentences to contain at least 4 words each
    # This ensures BLEU-4 is calculable (not 0.0 due to short length) when grouped by type
    preds = ["this is a phrase", "this is a passage", "this is a multi"]
    refs = ["this is a phrase", "this is a passage", "this is a multi"]
    spoiler_types = ["phrase", "passage", "multi"]
    results = compute_metrics_by_spoiler_type(preds, refs, spoiler_types)
    assert "all" in results
    assert "phrase" in results
    assert "passage" in results
    assert "multi" in results
    assert results["phrase"]["bleu4"] > 99.0
