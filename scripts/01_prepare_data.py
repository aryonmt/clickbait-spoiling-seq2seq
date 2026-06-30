import argparse
import os
from collections import Counter

from spoiler_gen.config import load_config
from spoiler_gen.data.loader import load_clickbait_split
from spoiler_gen.data.preprocess import preprocess_split, save_processed
from spoiler_gen.utils.logging_utils import get_logger

logger = get_logger("01_prepare_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Webis raw clickbait data into processed train format."
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Path to the directory containing configuration yaml files.",
    )
    return parser.parse_args()


def analyze_and_log_distribution(records: list[dict], split_name: str) -> None:
    """Log split size and verify spoiler-type distributions against SemEval paper targets.

    Targets: ~43% phrase, ~40% passage, ~17% multi.
    Logs warning if any class deviates by more than 5%.
    """
    total = len(records)
    if total == 0:
        logger.warning(f"Split '{split_name}' contains zero records.")
        return

    counts = Counter(r["spoiler_type"] for r in records)
    logger.info(f"--- Distribution Stats for [{split_name}] (Total: {total}) ---")

    # Target proportions in TohokuNLP paper (Table 1 / Section 5)
    targets = {"phrase": 0.43, "passage": 0.40, "multi": 0.17}

    for key, count in counts.items():
        proportion = count / total
        logger.info(f"  - {key}: {count} samples ({proportion:.2%})")

        # Check deviance for standard classes
        if key in targets:
            target = targets[key]
            deviation = abs(proportion - target)
            if deviation > 0.05:
                logger.warning(
                    f"Spoiler type '{key}' in split '{split_name}' deviates from "
                    f"paper target ({target:.2%}) by {deviation:.2%} (observed: {proportion:.2%}). "
                    "This might indicate parsing discrepancies or a custom dataset split."
                )


def main() -> None:
    args = parse_args()
    logger.info(f"Loading configuration from directory: {args.config_dir}")
    cfg = load_config(args.config_dir)

    # Process each configured split
    splits_to_process = [
        ("train", cfg.data.train_file),
        ("validation", cfg.data.val_file),
    ]

    # Test file is optional
    if cfg.data.test_file:
        splits_to_process.append(("test", cfg.data.test_file))

    for split_name, file_name in splits_to_process:
        raw_path = os.path.join(cfg.data.raw_dir, file_name)
        processed_path = os.path.join(cfg.data.processed_dir, file_name)

        logger.info(f"Processing split '{split_name}' from raw path: {raw_path}")

        try:
            # 1. Load
            samples = load_clickbait_split(raw_path)

            # 2. Preprocess
            processed_records = preprocess_split(
                samples=samples,
                max_input_length=cfg.data.max_input_length,
                template=cfg.data.input_template,
                join_token=cfg.data.spoiler_join_token,
            )

            # 3. Analyze
            analyze_and_log_distribution(processed_records, split_name)

            # 4. Save
            save_processed(processed_records, processed_path)

        except FileNotFoundError as e:
            logger.error(f"Failed to process split '{split_name}': {e}")
            if split_name != "test":
                # Only fail hard if train or validation splits are missing
                raise e
            else:
                logger.warning("Optional test split processing skipped due to missing file.")

    logger.info("Data preparation pipeline step completed successfully.")


if __name__ == "__main__":
    main()
