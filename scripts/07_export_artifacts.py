import argparse
import os
import zipfile

from spoiler_gen.config import load_config
from spoiler_gen.utils.logging_utils import get_logger

logger = get_logger("07_export_artifacts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export small project artifacts to zip.")
    parser.add_argument("--config-dir", type=str, default="configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config_dir)

    export_dir = os.path.join(cfg.base.output_root, "export")
    os.makedirs(export_dir, exist_ok=True)

    zip_path = os.path.join(export_dir, "clickbait_spoiling_results.zip")
    logger.info(f"Creating export zip package at: {zip_path}")

    pred_dir = os.path.join(cfg.base.output_root, "predictions")
    report_dir = os.path.join(cfg.base.output_root, "reports")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Add reports
        if os.path.exists(report_dir):
            for root, _, files in os.walk(report_dir):
                for file in files:
                    fp = os.path.join(root, file)
                    arcname = os.path.join("reports", file)
                    zip_file.write(fp, arcname)
                    logger.info(f"Archived report: {arcname}")

        # Add predictions (including ensemble outputs)
        if os.path.exists(pred_dir):
            for root, _, files in os.walk(pred_dir):
                for file in files:
                    if "ensemble" in file or "seed43" in file:
                        fp = os.path.join(root, file)
                        arcname = os.path.join("predictions", file)
                        zip_file.write(fp, arcname)
                        logger.info(f"Archived prediction file: {arcname}")

    logger.info("Artifacts export packaging completed successfully.")


if __name__ == "__main__":
    main()
