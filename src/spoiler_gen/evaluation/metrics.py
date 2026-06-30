import numpy as np
import sacrebleu
import evaluate
import nltk
from spoiler_gen.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Preemptively download NLTK packages required for METEOR evaluation
# This avoids runtime errors when internet connectivity is disabled on Kaggle
try:
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
except Exception as e:
    logger.warning(
        f"Failed to preemptively download NLTK data: {e}. METEOR evaluation might fail if offline."
    )


def compute_bleu4(predictions: list[str], references: list[str]) -> float:
    """Compute the corpus-level BLEU-4 score using sacrebleu.

    BLEU score returned is on a 0-100 scale.

    Args:
        predictions: List of generated spoiler prediction strings.
        references: List of gold spoiler target strings.

    Returns:
        BLEU-4 score as a float (0.0 to 100.0).
    """
    if not predictions or not references:
        return 0.0
    # sacrebleu expects a list of reference lists for each candidate translation
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return float(bleu.score)


def compute_bertscore(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute token-level similarity using contextual BERT embeddings (BERTScore).

    Args:
        predictions: List of generated spoiler prediction strings.
        references: List of gold spoiler target strings.

    Returns:
        A dictionary with precision, recall, and f1 scores (scaled 0.0 to 1.0).
    """
    if not predictions or not references:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    try:
        bertscore_metric = evaluate.load("bertscore")
        results = bertscore_metric.compute(
            predictions=predictions, references=references, lang="en"
        )
        return {
            "precision": float(np.mean(results["precision"])),
            "recall": float(np.mean(results["recall"])),
            "f1": float(np.mean(results["f1"])),
        }
    except Exception as e:
        logger.error(f"Failed to compute BERTScore: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def compute_meteor(predictions: list[str], references: list[str]) -> float:
    """Compute the METEOR score using evaluate library (NLTK METEOR wrapper).

    Please note that this is backed by NLTK's METEOR engine, which may differ
    slightly from Java-based METEOR 1.5 used in original SemEval evaluation.

    Args:
        predictions: List of generated spoiler prediction strings.
        references: List of gold spoiler target strings.

    Returns:
        METEOR score as a float (0.0 to 1.0).
    """
    if not predictions or not references:
        return 0.0

    try:
        meteor_metric = evaluate.load("meteor")
        results = meteor_metric.compute(predictions=predictions, references=references)
        return float(results["meteor"])
    except Exception as e:
        logger.error(f"Failed to compute METEOR: {e}")
        return 0.0


def compute_all_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute BLEU-4, BERTScore, and METEOR.

    Args:
        predictions: List of generated spoiler prediction strings.
        references: List of gold spoiler target strings.

    Returns:
        A dictionary of all computed metrics.
    """
    bleu = compute_bleu4(predictions, references)
    bert = compute_bertscore(predictions, references)
    meteor = compute_meteor(predictions, references)

    return {
        "bleu4": bleu,
        "bertscore_p": bert["precision"],
        "bertscore_r": bert["recall"],
        "bertscore_f1": bert["f1"],
        "meteor": meteor,
    }


def compute_metrics_by_spoiler_type(
    predictions: list[str], references: list[str], spoiler_types: list[str]
) -> dict[str, dict[str, float]]:
    """Compute all evaluation metrics overall and broken down by spoiler type.

    Args:
        predictions: List of generated spoiler prediction strings.
        references: List of gold spoiler target strings.
        spoiler_types: List of tags associated with each example (e.g. 'phrase', 'passage', 'multi').

    Returns:
        A dictionary mapping setting names ('all', 'phrase', 'passage', 'multi')
        to their respective metric results.
    """
    if len(predictions) != len(references) or len(predictions) != len(spoiler_types):
        raise ValueError("Predictions, references, and spoiler_types must have equal lengths.")

    # Compute overall ("all") metrics first
    logger.info("Computing metrics overall ('all')...")
    results = {"all": compute_all_metrics(predictions, references)}

    # Group indices by spoiler type
    type_groups = {}
    for idx, s_type in enumerate(spoiler_types):
        type_groups.setdefault(s_type, []).append(idx)

    for s_type, indices in type_groups.items():
        logger.info(f"Computing metrics for spoiler type '{s_type}' ({len(indices)} samples)...")
        sub_preds = [predictions[i] for i in indices]
        sub_refs = [references[i] for i in indices]
        results[s_type] = compute_all_metrics(sub_preds, sub_refs)

    return results
