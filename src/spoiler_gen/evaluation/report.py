import os
import pandas as pd
from spoiler_gen.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_report(metrics_by_setting: dict[str, dict[str, dict[str, float]]]) -> pd.DataFrame:
    """Format nested metrics dictionary into a tidy Pandas DataFrame for reporting.

    Args:
        metrics_by_setting: Nested dictionary structured as:
            {
                "setting_name": {
                    "spoiler_type": {
                        "bleu4": 42.1,
                        "bertscore_f1": 0.85,
                        "meteor": 0.35,
                        ...
                    }
                }
            }

    Returns:
        A tidy Pandas DataFrame containing columns:
            ['Setting', 'Spoiler Type', 'BLEU-4', 'BERTScore F1', 'METEOR']
    """
    rows = []
    for setting_name, type_metrics in metrics_by_setting.items():
        for spoiler_type, metrics in type_metrics.items():
            rows.append(
                {
                    "Setting": setting_name,
                    "Spoiler Type": spoiler_type,
                    "BLEU-4": round(metrics.get("bleu4", 0.0), 2),
                    "BERTScore F1": round(metrics.get("bertscore_f1", 0.0), 4),
                    "METEOR": round(metrics.get("meteor", 0.0), 4),
                }
            )
    return pd.DataFrame(rows)


def save_report(df: pd.DataFrame, path_prefix: str) -> None:
    """Save the report DataFrame to both CSV and a pretty Markdown table.

    Args:
        df: The report DataFrame.
        path_prefix: Base file path (without extension, e.g., 'outputs/reports/eval_report').
    """
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)

    csv_path = f"{path_prefix}.csv"
    md_path = f"{path_prefix}.md"

    # Save CSV
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved evaluation report CSV to: {csv_path}")

    # Save Markdown table
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Clickbait Spoiling Evaluation Report\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n")
        logger.info(f"Saved evaluation report Markdown to: {md_path}")
    except Exception as e:
        logger.error(f"Failed to write markdown table: {e}")
