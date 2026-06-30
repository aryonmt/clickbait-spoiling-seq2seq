import argparse

from spoiler_gen.config import load_config
from spoiler_gen.inference.generate import run_inference_for_seed
from spoiler_gen.utils.logging_utils import get_logger

logger = get_logger("04_generate_predictions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions from trained seed models.")
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["validation", "test"],
        help="Split to run predictions on.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing configuration files.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[43, 45, 46, 47, 48],
        help="Seeds of trained models to generate predictions for.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config_dir)

    logger.info(f"Running inference on split: {args.split} for seeds: {args.seeds}")

    for seed in args.seeds:
        try:
            run_inference_for_seed(seed, args.split, cfg)
        except Exception as e:
            logger.error(f"Error generating predictions for seed {seed}: {e}")
            raise e

    logger.info("Inference generation step completed successfully.")


if __name__ == "__main__":
    main()
