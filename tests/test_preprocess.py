from spoiler_gen.data.preprocess import build_input_text, build_target_text
from spoiler_gen.data.schema import ClickbaitSample


def test_build_input_text():
    sample = ClickbaitSample(
        uuid="1",
        post_text=["What happened?"],
        target_title="Shocking",
        target_paragraphs=["This is paragraph one.", "This is paragraph two."],
        spoiler=["Nothing"],
        tags=["phrase"],
    )
    template = "question: {post_text} context: {target_title} - {target_paragraphs}"
    inp = build_input_text(sample, template, max_input_length=100)
    assert "question: What happened?" in inp
    assert "context: Shocking -" in inp


def test_build_target_text():
    sample = ClickbaitSample(
        uuid="1",
        post_text=["What happened?"],
        target_title="Shocking",
        target_paragraphs=["P1"],
        spoiler=["Nothing", "Special"],
        tags=["multi"],
    )
    target = build_target_text(sample, join_token=" ")
    assert target == "Nothing Special"
