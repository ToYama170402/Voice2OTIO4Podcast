import pytest

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


def _word(text: str) -> list[WordTimestamp]:
    return [WordTimestamp(text, 0.0, 1.0, 1.0)]


def test_default_boundary_recognizes_sentence_punctuation() -> None:
    from podcastpreprocessor.segmentator import _default_boundary

    for text in (
        "次の文、",
        "終わり。",
        "本当？",
        "すごい！",
        "yes,",
        "end.",
        "why?",
        "wow!",
    ):
        assert _default_boundary(_word(text)), text


def test_default_boundary_recognizes_closing_parts_of_speech() -> None:
    from podcastpreprocessor.segmentator import _default_boundary

    for text in ("彼は", "行った", "走る", "高い", "ゆっくり"):
        assert _default_boundary(_word(text)), text
    for text in ("東京", "食べ物", "字幕テキスト"):
        assert not _default_boundary(_word(text)), text
    assert not _default_boundary([])
    assert not _default_boundary(_word("   "))


class _FakeMorpheme:
    def __init__(self, pos: tuple[str, ...], surface: str = "") -> None:
        self._pos = pos
        self._surface = surface

    def part_of_speech(self) -> tuple[str, ...]:
        return self._pos

    def surface(self) -> str:
        return self._surface


def _morph(
    major: str, sub: str = "一般", surface: str = "", conjugation: str = ""
) -> _FakeMorpheme:
    return _FakeMorpheme((major, sub, "", "", conjugation, ""), surface)


def test_is_bunsetsu_head_rules() -> None:
    from podcastpreprocessor.segmentator import _is_bunsetsu_head

    noun = _morph("名詞")
    # Function words never head a bunsetsu.
    assert not _is_bunsetsu_head(_morph("助詞"), noun)
    assert not _is_bunsetsu_head(_morph("助動詞"), noun)
    # Noun chains: a noun/pronoun after a noun-like word continues the unit.
    assert not _is_bunsetsu_head(_morph("名詞"), noun)
    assert not _is_bunsetsu_head(_morph("名詞"), _morph("代名詞"))
    assert not _is_bunsetsu_head(_morph("名詞"), _morph("連体詞"))
    assert not _is_bunsetsu_head(_morph("名詞"), _morph("接頭辞"))
    assert not _is_bunsetsu_head(_morph("代名詞"), noun)
    assert _is_bunsetsu_head(_morph("名詞"), _morph("動詞"))
    assert _is_bunsetsu_head(_morph("名詞"), _morph("助詞"))
    assert _is_bunsetsu_head(_morph("代名詞"), _morph("助詞"))
    # Other content parts of speech head their own unit.
    assert _is_bunsetsu_head(_morph("接頭辞"), noun)
    assert _is_bunsetsu_head(_morph("副詞"), _morph("助詞"))
    assert _is_bunsetsu_head(_morph("接続詞"), _morph("助詞"))
    assert _is_bunsetsu_head(_morph("感動詞"), _morph("助詞"))
    assert _is_bunsetsu_head(_morph("形状詞"), _morph("助詞"))


def test_is_bunsetsu_head_non_independent_verbs() -> None:
    from podcastpreprocessor.segmentator import _is_bunsetsu_head

    non_independent = _morph("動詞", "非自立可能", conjugation="上一段-ア行")
    assert not _is_bunsetsu_head(non_independent, _morph("動詞"))
    assert not _is_bunsetsu_head(non_independent, _morph("形容詞"))
    assert not _is_bunsetsu_head(non_independent, _morph("助詞", surface="て"))
    assert not _is_bunsetsu_head(non_independent, _morph("助詞", surface="で"))
    assert _is_bunsetsu_head(non_independent, _morph("助詞", surface="が"))
    # Sahen nouns absorb a following sahen verb; other nouns do not.
    sahen = _morph("動詞", "非自立可能", conjugation="サ行変格")
    assert not _is_bunsetsu_head(sahen, _morph("名詞"))
    assert _is_bunsetsu_head(non_independent, _morph("名詞"))
    # Ordinary verbs are unaffected, as are short POS tuples.
    assert _is_bunsetsu_head(_morph("動詞"), _morph("動詞"))
    short = _FakeMorpheme(("動詞", "非自立可能"))
    assert _is_bunsetsu_head(short, _morph("名詞"))
    non_independent_adj = _morph("形容詞", "非自立可能")
    assert not _is_bunsetsu_head(non_independent_adj, _morph("動詞"))


def test_segment_raw_validates_limits() -> None:
    source = _source([("字幕", 0.0, 0.8)])
    with pytest.raises(ValueError, match="segmentation limits must be positive"):
        segment_raw(source, max_characters=0)
    with pytest.raises(ValueError, match="segmentation limits must be positive"):
        segment_raw(source, max_duration=0.0)
    with pytest.raises(ValueError, match="segmentation limits must be positive"):
        segment_raw(source, preferred_duration=-1.0)
    with pytest.raises(
        ValueError, match="preferred_duration cannot exceed max_duration"
    ):
        segment_raw(source, max_duration=2.0, preferred_duration=3.0)
    with pytest.raises(ValueError, match="track_index must be positive"):
        segment_raw(source, track_index=0)


def test_segment_raw_without_words_fallback_details() -> None:
    source = RawSegment(2, 1.0, 2.0, "  音声なし  ", 0.3, source_id="src9")

    segments = segment_raw(source, track_index=3)

    assert len(segments) == 1
    segment = segments[0]
    assert segment.segment_id == "2_0"
    # The fallback always uses track 1 and the stripped source text.
    assert segment.track_index == 1
    assert segment.text == "音声なし"
    assert segment.start == 1.0
    assert segment.end == 2.0
    assert segment.no_speech_prob == 0.3
    assert segment.source_id == "src9"
    assert segment.words == []

    assert segment_raw(RawSegment(2, 1.0, 2.0, "   ", 0.0)) == []
    assert segment_raw(RawSegment(2, 1.0, 2.0, "", 0.0)) == []


def test_segment_raw_propagates_track_and_ids() -> None:
    source = _source(
        [
            ("これは", 0.0, 0.8),
            ("字幕", 0.8, 1.6),
            ("テキスト", 1.6, 2.4),
            ("です。", 2.4, 3.2),
        ],
        segment_id=5,
    )
    source.no_speech_prob = 0.42
    source.source_id = "src7"

    segments = segment_raw(
        source, max_characters=3, track_index=2, preferred_duration=0.1
    )

    assert [item.segment_id for item in segments] == ["5_0", "5_1"]
    assert all(item.track_index == 2 for item in segments)
    assert all(item.no_speech_prob == 0.42 for item in segments)
    assert all(item.source_id == "src7" for item in segments)


def test_segment_raw_preferred_duration_boundary() -> None:
    # Units are これは and 字幕テキストです。 (です attaches to its unit).
    source = _source(
        [
            ("これは", 0.0, 2.0),
            ("字幕", 2.0, 2.3),
            ("テキスト", 2.3, 2.6),
            ("です。", 2.6, 3.0),
        ]
    )

    segments = segment_raw(
        source,
        max_characters=100,
        max_duration=4.0,
        preferred_duration=2.0,
        boundary_predicate=lambda words: True,
    )
    assert [item.text for item in segments] == ["これは", "字幕テキストです。"]

    segments = segment_raw(
        source,
        max_characters=100,
        max_duration=4.0,
        preferred_duration=2.01,
        boundary_predicate=lambda words: True,
    )
    assert [item.text for item in segments] == ["これは字幕テキストです。"]


def test_segment_raw_max_characters_boundary() -> None:
    source = _source(
        [
            ("これは", 0.0, 0.2),
            ("字幕", 0.2, 0.4),
            ("テキスト", 0.4, 0.6),
            ("です。", 0.6, 0.8),
        ]
    )

    segments = segment_raw(source, max_characters=12, preferred_duration=3.5)
    assert [item.text for item in segments] == ["これは字幕テキストです。"]

    segments = segment_raw(source, max_characters=11, preferred_duration=3.5)
    assert [item.text for item in segments] == ["これは", "字幕テキストです。"]


def test_segment_source_assigns_probability_by_first_word() -> None:
    first = _source([("今日は", 0.0, 0.6), ("ちょ", 0.6, 0.9)], segment_id=7)
    first.no_speech_prob = 0.1
    second = _source([("っと", 0.9, 1.2), ("思います", 1.2, 2.0)], segment_id=8)
    second.no_speech_prob = 0.9

    segments = segment_source([first, second], max_characters=5)

    assert [item.text for item in segments] == ["今日は", "ちょっと", "思います"]
    assert [item.no_speech_prob for item in segments] == [0.1, 0.1, 0.9]
    assert all(item.source_id == first.source_id for item in segments)


def test_segment_source_edge_cases() -> None:
    assert segment_source([]) == []

    worded = _source([("今日は", 0.0, 0.6)], segment_id=3)
    wordless = RawSegment(4, 0.6, 1.6, " 補足 ", 0.0)
    segments = segment_source([worded, wordless])
    assert [item.text for item in segments] == ["今日は", "補足"]


def test_filler_char_spans_grouping() -> None:
    from sudachipy import SplitMode

    from podcastpreprocessor.segmentator import _TOKENIZER, _filler_char_spans

    morphemes = _TOKENIZER.tokenize("えーと、今日は", SplitMode.A)
    assert _filler_char_spans(morphemes) == [(0, 4)]

    morphemes = _TOKENIZER.tokenize("えーとですね", SplitMode.A)
    assert _filler_char_spans(morphemes) == [(0, 3)]

    morphemes = _TOKENIZER.tokenize("今日は晴れです", SplitMode.A)
    assert _filler_char_spans(morphemes) == []


def test_segment_source_detects_filler_beyond_tokenize_hard_limit() -> None:
    phrase = "今日はいい天気ですね。"
    words = []
    t = 0.0
    for _ in range(1000):
        for ch in phrase:
            words.append(WordTimestamp(ch, t, t + 0.09, 1.0))
            t += 0.1
    words.append(WordTimestamp("えーと", t, t + 0.3, 1.0))
    words.append(WordTimestamp("次です", t + 0.3, t + 0.6, 1.0))
    source = RawSegment(9, 0.0, t + 0.6, "", 0.0, words)

    segments = segment_source([source])

    assert any(item.text == "えーと" for item in segments)
