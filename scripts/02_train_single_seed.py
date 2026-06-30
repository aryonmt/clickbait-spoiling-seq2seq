import argparse
import os

from spoiler_gen.config import load_config
from spoiler_gen.data.dataset import ClickbaitSeq2SeqDataset, build_hf_dataset
from spoiler_gen.modeling.seq2seq_model import load_model_and_tokenizer
from spoiler_gen.training.trainer import train_one_seed
from spoiler_gen.utils.logging_utils import get_logger

logger = get_logger("02_train_single_seed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model for a single seed.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed to train on.")
    parser.add_argument("--config-dir", type=str, default="configs", help="Config directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config_dir)

    logger.info(f"Preparing datasets for training seed {args.seed}...")
    # Load tokenizer to build the datasets
    _, tokenizer = load_model_and_tokenizer(cfg.train.model_name_or_path)

    train_path = os.path.join(cfg.data.processed_dir, cfg.data.train_file)
    val_path = os.path.join(cfg.data.processed_dir, cfg.data.val_file)

    train_hf = build_hf_dataset(train_path, tokenizer, cfg.data)
    val_hf = build_hf_dataset(val_path, tokenizer, cfg.data)

    train_ds = ClickbaitSeq2SeqDataset(train_hf)
    val_ds = ClickbaitSeq2SeqDataset(val_hf)

    # Train the seed
    train_one_seed(args.seed, cfg, train_ds, val_ds)
    logger.info(f"Training for seed {args.seed} finished successfully.")


if __name__ == "__main__":
    main()
