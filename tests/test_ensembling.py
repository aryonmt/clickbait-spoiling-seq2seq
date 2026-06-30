import pytest

from spoiler_gen.modeling.ensembling import ensemble_predictions, select_ensemble_output


def test_select_ensemble_output():
    # Test the Kobayashi medoid selection with ties
    candidates = ["take a nap", "sleep", "take a nap", "taking a nap"]
    selected = select_ensemble_output(candidates)
    assert selected == "take a nap"


def test_ensemble_predictions_mismatch():
    per_seed = {
        "43": ["a", "b"],
        "45": ["a"],  # mismatched length
    }
    with pytest.raises(ValueError):
        ensemble_predictions(per_seed)
