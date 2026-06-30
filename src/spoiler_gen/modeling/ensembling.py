import Levenshtein


def edit_distance(a: str, b: str) -> int:
    """Calculate the Levenshtein edit distance between two strings.

    Args:
        a: First string.
        b: Second string.

    Returns:
        The Levenshtein edit distance as an integer.
    """
    return Levenshtein.distance(a, b)


def select_ensemble_output(candidates: list[str]) -> str:
    """Select the candidate with the smallest summed edit distance to all
    other candidates (post-hoc ensembling, Algorithm 1 / Kobayashi 2018).

    Args:
        candidates: predicted spoiler strings, one per seed model, for the
            SAME input example.

    Returns:
        The selected "medoid" string. If `candidates` has length 1, returns
        it unchanged. Ties are broken by earliest index (stable).
    """
    if not candidates:
        raise ValueError("Candidate list cannot be empty.")
    if len(candidates) == 1:
        return candidates[0]

    # Calculate sum of edit distances for each candidate to all other candidates
    min_dist_sum = float("inf")
    best_candidate = candidates[0]

    for candidate in candidates:
        current_sum = 0
        for other in candidates:
            if candidate != other:
                current_sum += edit_distance(candidate, other)

        # Select minimum sum. Strict inequality preserves stable tie-breaking
        if current_sum < min_dist_sum:
            min_dist_sum = current_sum
            best_candidate = candidate

    return best_candidate


def ensemble_predictions(per_seed_predictions: dict[str, list[str]]) -> list[str]:
    """Combine predictions from multiple seeds using edit-distance ensembling.

    Args:
        per_seed_predictions: Dictionary mapping seed IDs to lists of prediction strings.
            The lists must be perfectly aligned (same length and order).

    Returns:
        A list of combined system predictions.

    Raises:
        ValueError: If seed prediction lists have mismatched lengths or if dict is empty.
    """
    if not per_seed_predictions:
        raise ValueError("Prediction dictionary cannot be empty.")

    seeds = list(per_seed_predictions.keys())
    num_examples = len(per_seed_predictions[seeds[0]])

    # Validation: Mismatched lengths
    for seed in seeds:
        if len(per_seed_predictions[seed]) != num_examples:
            raise ValueError(
                f"Mismatched prediction lengths. Seed '{seed}' has length {len(per_seed_predictions[seed])} "
                f"but seed '{seeds[0]}' has length {num_examples}."
            )

    final_predictions = []
    for i in range(num_examples):
        # Extract candidate list for example i across all seeds
        candidates = [per_seed_predictions[seed][i] for seed in seeds]
        selected = select_ensemble_output(candidates)
        final_predictions.append(selected)

    return final_predictions
