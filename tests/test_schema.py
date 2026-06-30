from spoiler_gen.data.schema import ClickbaitSample


def test_clickbait_sample_from_raw_dict():
    # Test camelCase mapping and single string postText conversion to list
    raw = {
        "uuid": "abc",
        "postText": "Single post text string",
        "targetTitle": "My Title",
        "targetParagraphs": ["Paragraph 1"],
        "spoiler": "Single spoiler",
        "tags": "phrase",
    }
    sample = ClickbaitSample.from_raw_dict(raw)
    assert sample.uuid == "abc"
    assert sample.post_text == ["Single post text string"]
    assert sample.target_title == "My Title"
    assert sample.target_paragraphs == ["Paragraph 1"]
    assert sample.spoiler == ["Single spoiler"]
    assert sample.spoiler_type == "phrase"
