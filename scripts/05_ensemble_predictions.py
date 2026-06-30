import argparse
import os

from spoiler_gen.config import load_config
from spoiler_gen.modeling.ensembling import ensemble_predictions
from spoiler_gen.utils.io_utils import read_jsonl, write_jsonl
from spoiler_gen.utils.logging_utils import get_logger

logger = get_logger("05_ensemble_predictions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble multi-seed predictions.")
    parser.add_argument("--split", type=str, required=True, choices=["validation", "test"])
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 45, 46, 47, 48])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config_dir)

    logger.info(f"Loading predictions for ensembling ({args.split})...")

    per_seed_preds = {}
    pred_dir = os.path.join(cfg.base.output_root, "predictions")

    # Load each seed's prediction file
    uuids_order = None
    metadata = {}  # To keep track of spoiler types and align them

    for seed in args.seeds:
        file_path = os.path.join(pred_dir, f"seed{seed}_{args.split}.jsonl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing prediction file for seed {seed} at: {file_path}")

        records = read_jsonl(file_path)

        # Check alignment / Sort by uuid to be defensive
        records.sort(key=lambda x: x["uuid"])

        current_uuids = [r["uuid"] for r in records]
        if uuids_order is None:
            uuids_order = current_uuids
            for r in records:
                metadata[r["uuid"]] = r["spoiler_type"]
        else:
            if current_uuids != uuids_order:
                raise ValueError(f"UUID order alignment mismatch in seed {seed} prediction file.")

        per_seed_preds[str(seed)] = [r["prediction"] for r in records]

    # Run ensembling medoid logic
    logger.info("Ensembling predictions using post-hoc edit distance medoids...")
    final_preds = ensemble_predictions(per_seed_preds)

    # Construct ensembled records
    ensembled_records = []
    for i, uuid in enumerate(uuids_order):
        ensembled_records.append(
            {"uuid": uuid, "prediction": final_preds[i], "spoiler_type": metadata[uuid]}
        )

    out_file = os.path.join(pred_dir, f"ensemble_{args.split}.jsonl")
    write_jsonl(ensembled_records, out_file)
    logger.info(f"Ensembled predictions saved to: {out_file}")


if __name__ == "__main__":
    main()
