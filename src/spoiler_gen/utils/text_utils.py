import re


def normalize_whitespace(text: str) -> str:
    """Replace multiple spaces/newlines with a single space and strip.

    Args:
        text: The input text to normalize.

    Returns:
        The normalized text with single spaces.
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def join_list_field(value: str | list[str]) -> str:
    """Handle inconsistent corpus typing.

    Converts str or list[str] uniformly to a single string.

    Args:
        value: Input string or list of strings.

    Returns:
        A combined, normalized string.
    """
    if isinstance(value, list):
        return " ".join([v for v in value if v]).strip()
    if isinstance(value, str):
        return value.strip()
    return ""
