import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "outputs/logs") -> logging.Logger:
    """Configure and return a logger with both console and file handlers.

    Args:
        name: Name of the logger.
        log_dir: Directory where log files will be saved.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    # Stream Handler (Stdout for console/Kaggle)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler
    try:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback if directory creation or file writing fails
        logger.warning(f"Could not create file handler for logger {name}: {e}")

    return logger
