import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedModel, PreTrainedTokenizerBase
from spoiler_gen.config import AppConfig, TrainConfig
from spoiler_gen.utils.logging_utils import get_logger

logger = get_logger(__name__)


def generate_predictions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: torch.utils.data.Dataset,
    train_cfg: TrainConfig,
) -> list[str]:
    """Generate predictions for a given dataset using the trained model in batches.

    Args:
        model: The trained PreTrainedModel.
        tokenizer: The PreTrainedTokenizerBase.
        dataset: ClickbaitSeq2SeqDataset.
        train_cfg: TrainConfig containing beam search parameters.

    Returns:
        A list of decoded prediction strings.
    """
    model.eval()

    # Use standard HF DataCollatorForSeq2Seq for padding
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="max_length")
    dataloader = DataLoader(
        dataset, batch_size=train_cfg.per_device_eval_batch_size, collate_fn=collator, shuffle=False
    )

    predictions = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            # Move inputs to the same device as the model
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=train_cfg.generation_max_length,
                num_beams=train_cfg.generation_num_beams,
                early_stopping=True,
            )

            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # Normalize and strip spaces
            predictions.extend([p.strip() for p in decoded_preds])

    return predictions


def run_inference_for_seed(seed: int, split: str, app_cfg: AppConfig) -> str:
    """Load seed checkpoint, run inference over a split, and save output JSONL.

    Args:
        seed: Model seed checkpoint to load.
        split: Data split ('validation' or 'test') to run on.
        app_cfg: AppConfig object.

    Returns:
        The file path of saved prediction JSONL.
    """
    from spoiler_gen.data.dataset import build_hf_dataset, ClickbaitSeq2SeqDataset
    from spoiler_gen.modeling.seq2seq_model import load_model_and_tokenizer
    from spoiler_gen.utils.io_utils import write_jsonl, read_jsonl

    # 1. Build Paths
    model_dir = os.path.join(
        app_cfg.base.output_root, "checkpoints", f"flan-t5-large-seed{seed}", "final_model"
    )
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Trained model checkpoint not found at: {model_dir}")

    file_name = app_cfg.data.val_file if split == "validation" else app_cfg.data.test_file
    if not file_name:
        raise ValueError(f"Config path for split '{split}' is not defined.")

    processed_split_path = os.path.join(app_cfg.data.processed_dir, file_name)
    if not os.path.exists(processed_split_path):
        raise FileNotFoundError(f"Processed split file not found at: {processed_split_path}")

    # 2. Load model and tokenizer
    logger.info(f"Loading trained model for seed {seed} from {model_dir}...")
    model, tokenizer = load_model_and_tokenizer(model_dir)

    # 3. Load and prepare dataset
    logger.info(f"Tokenizing split '{split}' using loader...")
    hf_ds = build_hf_dataset(processed_split_path, tokenizer, app_cfg.data)
    torch_ds = ClickbaitSeq2SeqDataset(hf_ds)

    # 4. Generate predictions
    logger.info(f"Generating predictions for seed {seed} on split '{split}'...")
    preds = generate_predictions(model, tokenizer, torch_ds, app_cfg.train)

    # 5. Load original processed lines to map with predictions
    raw_processed = read_jsonl(processed_split_path)
    if len(preds) != len(raw_processed):
        raise ValueError(
            f"Generated predictions ({len(preds)}) mismatched raw processed examples ({len(raw_processed)})"
        )

    # Map predictions to output format
    out_records = []
    for i, item in enumerate(raw_processed):
        out_records.append(
            {"uuid": item["uuid"], "prediction": preds[i], "spoiler_type": item["spoiler_type"]}
        )

    out_dir = os.path.join(app_cfg.base.output_root, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"seed{seed}_{split}.jsonl")

    write_jsonl(out_records, out_file)
    logger.info(f"Predictions saved to: {out_file}")

    return out_file
