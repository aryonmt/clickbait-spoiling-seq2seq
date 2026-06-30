from spoiler_gen.evaluation.metrics import compute_bleu4, compute_metrics_by_spoiler_type


def test_compute_bleu4():
    preds = ["this is a test", "hello world"]
    refs = ["this is a test", "hello world"]
    score = compute_bleu4(preds, refs)
    assert score > 99.0


def test_compute_metrics_by_spoiler_type():
    preds = ["yes", "no", "maybe"]
    refs = ["yes", "no", "maybe"]
    spoiler_types = ["phrase", "passage", "multi"]
    results = compute_metrics_by_spoiler_type(preds, refs, spoiler_types)
    assert "all" in results
    assert "phrase" in results
    assert "passage" in results
    assert "multi" in results
    assert results["phrase"]["bleu4"] > 99.0
