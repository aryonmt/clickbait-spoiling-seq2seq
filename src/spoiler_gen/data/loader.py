import json
import os

from tqdm import tqdm

from spoiler_gen.data.schema import ClickbaitSample
from spoiler_gen.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file line-by-line with a progress bar.

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of raw dictionary objects.

    Raises:
        FileNotFoundError: If the file is missing, pointing to download instructions.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"The required data file was not found at: '{path}'. "
            f"Please register at the PAN Webis clickbait challenge "
            f"(https://pan.webis.de/semeval23/pan23-web/clickbait-challenge.html) "
            f"and place your raw splits ('train.jsonl', 'validation.jsonl') in 'data/raw/'."
        )

    records = []
    logger.info(f"Loading raw lines from {path}...")

    # Pre-count lines for accurate progress bar
    try:
        with open(path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
    except Exception:
        total_lines = None

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc=f"Reading {os.path.basename(path)}"):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to decode JSON line: {e}. Skipping line.")
    return records


def load_clickbait_split(path: str) -> list[ClickbaitSample]:
    """Load a raw JSONL split and convert it to ClickbaitSample objects.

    Skips and logs any malformed rows instead of crashing.

    Args:
        path: Path to the JSONL split file.

    Returns:
        A list of valid ClickbaitSample objects.
    """
    raw_records = load_jsonl(path)
    valid_samples = []
    malformed_count = 0

    for record in raw_records:
        try:
            sample = ClickbaitSample.from_raw_dict(record)
            valid_samples.append(sample)
        except Exception as e:
            malformed_count += 1
            logger.debug(
                f"Skipping malformed sample: {e}. Record ID: {record.get('uuid', 'unknown')}"
            )

    if malformed_count > 0:
        logger.warning(
            f"Skipped {malformed_count} malformed rows out of {len(raw_records)} in '{path}'."
        )
    else:
        logger.info(f"Successfully loaded {len(valid_samples)} samples from '{path}'.")

    return valid_samples
