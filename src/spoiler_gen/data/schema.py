from pydantic import BaseModel


class ClickbaitSample(BaseModel):
    uuid: str
    post_text: list[str]
    target_title: str
    target_paragraphs: list[str]
    spoiler: list[str]
    tags: list[str]

    @property
    def spoiler_type(self) -> str:
        """Return the first spoiler tag as the type, or 'unknown' if not present."""
        return self.tags[0] if self.tags else "unknown"

    @classmethod
    def from_raw_dict(cls, d: dict) -> "ClickbaitSample":
        """Parse raw Webis Clickbait Spoiling corpus dict.

        Tolerates camelCase (Webis original) and snake_case field names.

        Args:
            d: Raw dictionary from the JSONL lines.

        Returns:
            A validated ClickbaitSample instance.
        """
        # Parse postText field (can be a string or list of strings in the wild)
        post_text_raw = d.get("postText", d.get("post_text", []))
        if isinstance(post_text_raw, str):
            post_text = [post_text_raw]
        else:
            post_text = [str(t) for t in post_text_raw]

        # Parse targetParagraphs field
        target_paragraphs_raw = d.get("targetParagraphs", d.get("target_paragraphs", []))
        if isinstance(target_paragraphs_raw, str):
            target_paragraphs = [target_paragraphs_raw]
        else:
            target_paragraphs = [str(p) for p in target_paragraphs_raw]

        # Parse spoiler field
        spoiler_raw = d.get("spoiler", [])
        if isinstance(spoiler_raw, str):
            spoiler = [spoiler_raw]
        else:
            spoiler = [str(s) for s in spoiler_raw]

        # Parse tags field
        tags_raw = d.get("tags", [])
        if isinstance(tags_raw, str):
            tags = [tags_raw]
        else:
            tags = [str(t) for t in tags_raw]

        return cls(
            uuid=str(d.get("uuid", "")),
            post_text=post_text,
            target_title=str(d.get("targetTitle", d.get("target_title", ""))),
            target_paragraphs=target_paragraphs,
            spoiler=spoiler,
            tags=tags,
        )
