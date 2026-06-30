import json
import os
from typing import Any

import yaml


def read_yaml(path: str) -> dict[str, Any]:
    """Read and parse a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        A dictionary containing the parsed YAML data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a JSONL file line-by-line.

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of dictionaries.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSONL file not found at {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: str) -> None:
    """Write a list of dictionaries to a JSONL file.

    Args:
        records: List of dictionaries to write.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_dir_size_gb(path: str) -> float:
    """Calculate the total size of a directory in gigabytes (GB).

    Args:
        path: Path to the directory.

    Returns:
        The total size in GB.
    """
    total_size = 0
    if not os.path.exists(path):
        return 0.0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):  # Avoid broken symlinks
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)
