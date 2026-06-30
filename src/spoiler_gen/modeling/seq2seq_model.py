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

    This function wraps the loading process and enforces consistent torch_dtype
    and device map settings (such as fp16 and auto device mapping) to ensure
    reproducibility across runs (e.g. Kaggle T4 GPUs).

    The model defaults to 'google/flan-t5-large', which corresponds to the
    optimal base model architecture used in TohokuNLP's paper.

    Args:
        model_name_or_path: Path or identifier (e.g., 'google/flan-t5-large').

    Returns:
        A tuple of (PreTrainedModel, PreTrainedTokenizerBase).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    # Enforce consistent dtype for float16 operations
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    return model, tokenizer
