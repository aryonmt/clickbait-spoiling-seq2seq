import argparse
import os

from spoiler_gen.config import load_config
from spoiler_gen.evaluation.metrics import compute_metrics_by_spoiler_type
from spoiler_gen.evaluation.report import build_report, save_report
from spoiler_gen.utils.io_utils import read_jsonl
from spoiler_gen.utils.logging_utils import get_logger

logger = get_logger("06_evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions.")
    parser.add_argument("--split", type=str, required=True, choices=["validation", "test"])
    parser.add_argument(
        "--predictions-path", type=str, required=True, help="Path to predictions JSONL to evaluate."
    )
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 45, 46, 47, 48])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config_dir)

    # Load gold references
    gold_file = cfg.data.val_file if args.split == "validation" else cfg.data.test_file
    gold_path = os.path.join(cfg.data.processed_dir, gold_file)
    gold_records = read_jsonl(gold_path)
    gold_records.sort(key=lambda x: x["uuid"])

    gold_spoilers = [g["target_text"] for g in gold_records]
    uuids = [g["uuid"] for g in gold_records]
    spoiler_types = [g["spoiler_type"] for g in gold_records]

    # Load evaluated predictions
    pred_records = read_jsonl(args.predictions_path)
    pred_records.sort(key=lambda x: x["uuid"])

    # Validation
    pred_uuids = [p["uuid"] for p in pred_records]
    if pred_uuids != uuids:
        raise ValueError("Prediction and Gold alignment mismatch.")

    pred_spoilers = [p["prediction"] for p in pred_records]

    # Evaluate setting
    logger.info(f"Evaluating {args.predictions_path} against gold...")
    ensemble_results = compute_metrics_by_spoiler_type(pred_spoilers, gold_spoilers, spoiler_types)

    # Compute Single Baseline metrics (average across 5 individual seeds)
    logger.info("Computing Single Baseline metrics (average across 5 individual seeds)...")
    pred_dir = os.path.join(cfg.base.output_root, "predictions")

    single_seed_metrics_list = []
    for seed in args.seeds:
        seed_file = os.path.join(pred_dir, f"seed{seed}_{args.split}.jsonl")
        if os.path.exists(seed_file):
            s_records = read_jsonl(seed_file)
            s_records.sort(key=lambda x: x["uuid"])
            s_spoilers = [s["prediction"] for s in s_records]
            s_metrics = compute_metrics_by_spoiler_type(s_spoilers, gold_spoilers, spoiler_types)
            single_seed_metrics_list.append(s_metrics)

    # Average metrics over single seeds
    single_avg_results = {}
    if single_seed_metrics_list:
        spoiler_categories = ["all", "phrase", "passage", "multi"]
        metric_keys = ["bleu4", "bertscore_p", "bertscore_r", "bertscore_f1", "meteor"]

        for cat in spoiler_categories:
            single_avg_results[cat] = {}
            for metric in metric_keys:
                vals = [s_run[cat][metric] for s_run in single_seed_metrics_list if cat in s_run]
                single_avg_results[cat][metric] = sum(vals) / len(vals) if vals else 0.0

    # Compile into settings report
    metrics_by_setting = {"Single (Avg)": single_avg_results, "Ensemble": ensemble_results}

    report_df = build_report(metrics_by_setting)

    out_prefix = os.path.join(cfg.base.output_root, "reports", f"clickbait_evaluation_{args.split}")
    save_report(report_df, out_prefix)


if __name__ == "__main__":
    main()
