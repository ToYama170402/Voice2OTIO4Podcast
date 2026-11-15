from podcastpreprocessor.evaluator import evaluate_segment
from podcastpreprocessor.models import RawSegment, WordTimestamp
from podcastpreprocessor.segmentator import segment_raw, segment_source


def _source(words: list[tuple[str, float, float]], segment_id: int = 1) -> RawSegment:
    timestamps = [WordTimestamp(text, start, end, 1.0) for text, start, end in words]
    return RawSegment(
        segment_id, timestamps[0].start, timestamps[-1].end, "", 0.0, timestamps
    )


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


def test_segmentator_never_cuts_inside_a_bunsetsu_at_the_character_limit() -> None:
    source = _source(
        [
            ("字幕", 0.0, 0.8),
            ("テキスト", 0.8, 1.6),
            ("です。", 1.6, 2.4),
            ("次の", 2.4, 3.2),
            ("文章です", 3.2, 4.0),
        ],
        segment_id=3,
    )

    segments = segment_raw(source, max_characters=4)

    # 字幕テキストです。 is one bunsetsu chain for editing purposes: the
    # character limit must move the boundary, not split the compound noun.
    assert segments[0].text == "字幕テキストです。"
    assert [item.text for item in segments] == [
        "字幕テキストです。",
        "次の",
        "文章です",
    ]
    assert [word.word for item in segments for word in item.words] == [
        word.word for word in source.words
    ]


def test_segmentator_isolates_a_filler_at_the_start() -> None:
    source = _source(
        [
            ("えーと、", 0.0, 0.5),
            ("今日は", 0.5, 1.2),
            ("収録", 1.2, 1.9),
            ("です", 1.9, 2.4),
        ],
        segment_id=4,
    )

    segments = segment_raw(source)

    assert segments[0].text == "えーと、"
    assert all("えーと" not in item.text for item in segments[1:])
    assert evaluate_segment(segments[0]).is_cut_candidate
    assert not any(evaluate_segment(item).is_cut_candidate for item in segments[1:])


def test_segmentator_isolates_a_filler_in_the_middle() -> None:
    source = _source(
        [
            ("今日は", 0.0, 0.7),
            ("えーと", 0.7, 1.2),
            ("収録です", 1.2, 2.0),
        ],
        segment_id=5,
    )

    segments = segment_raw(source)

    assert [item.text for item in segments] == ["今日は", "えーと", "収録です"]


def test_segmentator_merges_a_filler_split_across_words() -> None:
    source = _source(
        [
            ("えー", 0.0, 0.3),
            ("と、", 0.3, 0.5),
            ("今日は", 0.5, 1.2),
            ("収録です", 1.2, 2.0),
        ],
        segment_id=6,
    )

    segments = segment_raw(source)

    assert segments[0].text == "えーと、"
    assert all("えー" not in item.text for item in segments[1:])


def test_segment_source_does_not_cut_at_raw_segment_edges() -> None:
    # The recognizer split ちょっと across two raw segments; segmenting each
    # raw segment on its own would slice the word in two.
    first = _source([("今日は", 0.0, 0.6), ("ちょ", 0.6, 0.9)], segment_id=7)
    second = _source([("っと", 0.9, 1.2), ("思います", 1.2, 2.0)], segment_id=8)

    segments = segment_source([first, second], max_characters=5)

    assert [item.text for item in segments] == ["今日は", "ちょっと", "思います"]


def test_segmentator_handles_missing_word_timestamps() -> None:
    source = RawSegment(2, 1.0, 2.0, "  音声なし  ", 0.0)

    segments = segment_raw(source)

    assert len(segments) == 1
    assert segments[0].text == "音声なし"


def test_segment_source_handles_text_beyond_sudachi_input_limit() -> None:
    # One hour of speech is ~50k characters; Sudachi rejects inputs over
    # 49149 bytes, so the merged source text must be tokenized in chunks.
    phrase = "今日はいい天気ですね。"
    words = []
    t = 0.0
    for _ in range(3000):
        for ch in phrase:
            words.append(WordTimestamp(ch, t, t + 0.09, 1.0))
            t += 0.1
    source = RawSegment(9, 0.0, t, phrase * 3000, 0.0, words)

    segments = segment_source([source])

    assert len(segments) > 100
    assert sum(len(item.words) for item in segments) == len(words)
    assert all(item.text for item in segments)
