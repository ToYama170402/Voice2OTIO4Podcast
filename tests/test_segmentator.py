from podcastpreprocessor.models import RawSegment, WordTimestamp
from podcastpreprocessor.segmentator import segment_raw


def test_segmentator_preserves_word_ranges_and_splits_at_limits() -> None:
    words = [
        WordTimestamp("これは", 0.0, 0.8, 1.0),
        WordTimestamp("字幕", 0.8, 1.6, 1.0),
        WordTimestamp("テキスト", 1.6, 2.4, 1.0),
        WordTimestamp("です。", 2.4, 3.2, 1.0),
        WordTimestamp("次の", 3.2, 4.0, 1.0),
        WordTimestamp("文章です", 4.0, 4.8, 1.0),
    ]
    source = RawSegment(1, 0.0, 4.8, "", 0.0, words)

    segments = segment_raw(source, max_characters=12, preferred_duration=2.0)

    assert [item.text for item in segments] == [
        "これは字幕テキストです。",
        "次の文章です",
    ]
    assert segments[0].start == words[0].start
    assert segments[-1].end == words[-1].end
    assert [word.word for item in segments for word in item.words] == [
        word.word for word in words
    ]


def test_segmentator_handles_missing_word_timestamps() -> None:
    source = RawSegment(2, 1.0, 2.0, "  音声なし  ", 0.0)

    segments = segment_raw(source)

    assert len(segments) == 1
    assert segments[0].text == "音声なし"
