import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def load_model_and_tokenizer(
    model_name_or_path: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load HuggingFace AutoModelForSeq2SeqLM and AutoTokenizer.

    This function wraps the loading process and enforces consistent
    device map settings to ensure reproducibility across runs.

    The model defaults to 'google/flan-t5-large', which corresponds to the
    optimal base model architecture used in TohokuNLP's paper.

    Args:
        model_name_or_path: Path or identifier (e.g., 'google/flan-t5-large').

    Returns:
        A tuple of (PreTrainedModel, PreTrainedTokenizerBase).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    # We load in default precision (float32). When fp16=True is set in training arguments,
    # Hugging Face Trainer automatically enables Automatic Mixed Precision (AMP).
    # AMP requires trainable parameters to be in float32 for numerical stability.
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path, device_map="auto" if torch.cuda.is_available() else None
    )

    return model, tokenizer
