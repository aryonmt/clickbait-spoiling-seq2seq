import datasets
import torch
from transformers import PreTrainedTokenizerBase

from spoiler_gen.config import DataConfig


def build_hf_dataset(
    processed_path: str, tokenizer: PreTrainedTokenizerBase, data_cfg: DataConfig
) -> datasets.Dataset:
    """Load JSONL from processed_path and return a tokenized HF dataset.

    Args:
        processed_path: Path to the processed JSONL file.
        tokenizer: The HuggingFace tokenizer.
        data_cfg: Configuration for data.

    Returns:
        A tokenized datasets.Dataset.
    """
    # Load JSON dataset using HuggingFace's datasets library
    dataset = datasets.load_dataset("json", data_files=processed_path, split="train")

    def tokenize_function(batch):
        # Tokenize inputs
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=data_cfg.max_input_length,
            truncation=True,
            padding="max_length",
        )

        # Tokenize targets (labels)
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=data_cfg.max_target_length,
            truncation=True,
            padding="max_length",
        )

        # Replace pad token id with -100 to ignore loss on pad tokens during training
        labels_ids = labels["input_ids"]
        adjusted_labels = []
        for item in labels_ids:
            adjusted_labels.append(
                [(-100 if token_id == tokenizer.pad_token_id else token_id) for token_id in item]
            )
        model_inputs["labels"] = adjusted_labels

        # Keep track of metadata for inference / evaluation
        model_inputs["uuid"] = batch["uuid"]
        model_inputs["spoiler_type"] = batch["spoiler_type"]
        return model_inputs

    # Map dataset tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset splits",
    )
    return tokenized_dataset


class ClickbaitSeq2SeqDataset(torch.utils.data.Dataset):
    """Torch dataset wrapper around HuggingFace's tokenized dataset."""

    def __init__(self, hf_dataset: datasets.Dataset):
        self.dataset = hf_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        # Convert lists to torch tensors where appropriate for the model input
        tensor_keys = ["input_ids", "attention_mask", "labels"]
        for key in tensor_keys:
            if key in item:
                item[key] = torch.tensor(item[key])
        return item
