from spoiler_gen.data.schema import ClickbaitSample
from spoiler_gen.utils.io_utils import write_jsonl
from spoiler_gen.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_input_text(sample: ClickbaitSample, template: str, max_input_length: int) -> str:
    """Build input text formatted as 'question: ... context: ...'.

    Truncates paragraphs in character space beforehand as a safe guard.

    Args:
        sample: The ClickbaitSample instance.
        template: Template string containing keys like {post_text}, {target_title}, {target_paragraphs}.
        max_input_length: Maximum allowed character length for the entire formatted input.

    Returns:
        The fully formatted template string.
    """
    post_text_str = " ".join(sample.post_text).strip()

    # Calculate character overhead from the empty template format
    empty_overhead = len(template.format(post_text="", target_title="", target_paragraphs=""))
    available_chars = (
        max_input_length - empty_overhead - len(post_text_str) - len(sample.target_title)
    )
    available_chars = max(0, available_chars)

    paragraphs_str = " ".join(sample.target_paragraphs).strip()
    if len(paragraphs_str) > available_chars:
        paragraphs_str = paragraphs_str[:available_chars].strip()

    return template.format(
        post_text=post_text_str, target_title=sample.target_title, target_paragraphs=paragraphs_str
    )


def build_target_text(sample: ClickbaitSample, join_token: str) -> str:
    """Concatenate multi-spoiler spans into a single target string.

    Args:
        sample: The ClickbaitSample instance.
        join_token: String separator (usually a space) to join multiple spoiler parts.

    Returns:
        A single concatenated target text string.
    """
    return join_token.join(sample.spoiler).strip()


def preprocess_split(
    samples: list[ClickbaitSample], max_input_length: int, template: str, join_token: str
) -> list[dict]:
    """Process a list of ClickbaitSample items into clean Seq2Seq train dictionary records.

    Args:
        samples: List of parsed clickbait samples.
        max_input_length: Maximum allowed character length.
        template: Formatted prompt template.
        join_token: Join separator for multi-target spoilers.

    Returns:
        A list of preprocessed records with keys: input_text, target_text, uuid, spoiler_type.
    """
    records = []
    for sample in samples:
        input_text = build_input_text(sample, template, max_input_length)
        target_text = build_target_text(sample, join_token)
        records.append(
            {
                "uuid": sample.uuid,
                "input_text": input_text,
                "target_text": target_text,
                "spoiler_type": sample.spoiler_type,
            }
        )
    return records


def save_processed(records: list[dict], path: str) -> None:
    """Save processed records using centralized IO utilities.

    Args:
        records: Preprocessed dictionary records.
        path: Output JSONL path.
    """
    write_jsonl(records, path)
    logger.info(f"Saved {len(records)} processed records to {path}.")
