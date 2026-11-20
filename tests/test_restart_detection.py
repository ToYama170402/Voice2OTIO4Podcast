from podcastpreprocessor.models import RawSegment, WordTimestamp
from podcastpreprocessor.pipeline import process_segments
from podcastpreprocessor.restart_detector import token_restart_regions


def char_words(text: str, start: float, step: float = 0.15) -> list[WordTimestamp]:
    """SenseVoice-style one-character tokens."""
    return [
        WordTimestamp(char, start + i * step, start + (i + 1) * step, 1.0)
        for i, char in enumerate(text)
    ]


def raw(segment_id: int, start: float, text: str, step: float = 0.15) -> RawSegment:
    words = char_words(text, start, step)
    end = words[-1].end if words else start
    return RawSegment(segment_id, start, end, text, 0.0, words)


def test_full_respeak_is_flagged_with_token_times() -> None:
    segments = [
        raw(1, 0.0, "本タブレット表示にすると"),
        raw(2, 2.1, "本タブレット表示にすると、こうなります。"),
    ]
    regions = token_restart_regions(segments)
    assert len(regions) == 1
    region = regions[0]
    assert region.start == 0.0
    assert abs(region.end - 1.8) < 1e-9
    assert region.text == "本タブレット表示にすると"
    assert region.reason == "言い直し (本タブレット表示にすると)"


def test_partial_respeak_of_incomplete_attempt_is_flagged() -> None:
    segments = [
        raw(1, 0.0, "野菜とか乗っけててる"),
        raw(2, 1.8, "野菜とか乗っけて焼くと美味しいです。"),
    ]
    regions = token_restart_regions(segments)
    assert len(regions) == 1
    assert regions[0].text.startswith("野菜とか乗っけて")


def test_complete_sentence_with_shared_opening_is_not_flagged() -> None:
    segments = [
        raw(1, 0.0, "私たちは昨日行きました。"),
        raw(2, 2.2, "私たちは今日行きます。"),
    ]
    assert token_restart_regions(segments) == []


def test_backchannel_is_not_flagged() -> None:
    segments = [
        raw(1, 0.0, "そうだね。"),
        raw(2, 1.0, "そうだね、じゃあ次に進もう。"),
    ]
    assert token_restart_regions(segments) == []


def test_filler_segment_is_not_flagged() -> None:
    segments = [
        raw(1, 0.0, "えーと"),
        raw(2, 0.8, "えーと、次は駅の話です。"),
    ]
    assert token_restart_regions(segments) == []


def test_large_gap_is_not_flagged() -> None:
    segments = [
        raw(1, 0.0, "本タブレット表示にすると"),
        raw(2, 12.0, "本タブレット表示にすると、こうなります。"),
    ]
    assert token_restart_regions(segments) == []


def test_within_repeat_region_covers_first_copy_only() -> None:
    segments = [raw(1, 0.0, "この辺はこの辺は多分スマホだと", step=0.12)]
    regions = token_restart_regions(segments)
    assert len(regions) == 1
    region = regions[0]
    assert region.start == 0.0
    assert abs(region.end - 0.48) < 1e-9
    assert region.text == "この辺は の繰り返し"


def test_emphatic_repetition_is_not_flagged() -> None:
    segments = [
        raw(1, 0.0, "そうそうそうそう。"),
        raw(2, 1.0, "なんかあれ、東日本と西日本で違ったよ"),
        raw(3, 3.0, "してないしてない。"),
        raw(4, 4.2, "うん。"),
    ]
    assert token_restart_regions(segments) == []


def test_segments_without_words_yield_no_regions() -> None:
    segments = [
        RawSegment(1, 0.0, 2.0, "本タブレット表示にすると", 0.0),
        RawSegment(2, 2.3, 5.0, "本タブレット表示にすると、こうなります。", 0.0),
    ]
    assert token_restart_regions(segments) == []


def test_pause_inside_long_token_still_pairs_attempts() -> None:
    # SenseVoice absorbs a pause into the previous token's duration:
    # the attempt's last token is long, then the re-speak follows in
    # the same raw segment.
    words = char_words("3列シートは", 0.0, step=0.15)
    words[-1] = WordTimestamp("は", words[-1].start, words[-1].start + 1.2, 1.0)
    pause_end = words[-1].end
    words += char_words("3列シートでワンセット", pause_end, step=0.15)
    segment = RawSegment(
        1, 0.0, words[-1].end, "3列シートは3列シートでワンセット", 0.0, words
    )
    regions = token_restart_regions([segment])
    assert len(regions) == 1
    assert regions[0].text.startswith("3列シート")


def test_process_segments_marks_pair_restart(tmp_path) -> None:
    segments = [
        raw(1, 0.0, "本タブレット表示にすると"),
        raw(2, 2.1, "本タブレット表示にすると、こうなります。"),
        raw(3, 5.4, "次の話題です。"),
    ]
    processed = process_segments(segments, "C:/audio/a.wav", str(tmp_path / "out.otio"))
    by_text = {segment.subtitle_text: segment for segment in processed}
    attempt = by_text["本タブレット表示にすると"]
    assert attempt.is_cut_candidate
    assert "言い直し" in attempt.marker_reason
    assert not by_text["次の話題です。"].is_cut_candidate


def test_restart_reason_is_added_to_existing_candidate() -> None:
    from podcastpreprocessor.evaluator import Evaluation
    from podcastpreprocessor.models import RefinedSegment
    from podcastpreprocessor.pipeline import _restart_pieces
    from podcastpreprocessor.restart_detector import RestartRegion

    words = char_words("この辺はこの辺は多分スマホだと", 0.0, step=0.12)
    refined = RefinedSegment(
        segment_id="source_1_0",
        track_index=1,
        start=0.0,
        end=words[-1].end,
        text="この辺はこの辺は多分スマホだと",
        no_speech_prob=0.0,
        words=words,
    )
    region = RestartRegion(0.0, 0.48, "この辺は の繰り返し")
    pieces = _restart_pieces(refined, [region], Evaluation(True, ("無音 1.0秒",)))
    assert len(pieces) == 1
    assert pieces[0][3].reasons == ("無音 1.0秒", "言い直し (この辺は の繰り返し)")


def test_process_segments_splits_within_segment_restart(tmp_path) -> None:
    segments = [raw(1, 0.0, "この辺はこの辺は多分スマホだと", step=0.12)]
    processed = process_segments(segments, "C:/audio/a.wav", str(tmp_path / "out.otio"))
    attempts = [
        segment
        for segment in processed
        if segment.is_cut_candidate and "言い直し" in segment.marker_reason
    ]
    assert len(attempts) == 1
    assert attempts[0].subtitle_text == "この辺は"
    # The pieces tile the whole segment: nothing is lost.
    total_frames = sum(segment.end_frame - segment.start_frame for segment in processed)
    assert total_frames == processed[-1].end_frame - processed[0].start_frame


def words_from_durations(
    text: str, start: float, durations: list[float]
) -> list[WordTimestamp]:
    words: list[WordTimestamp] = []
    cursor = start
    for char, duration in zip(text, durations, strict=True):
        words.append(WordTimestamp(char, cursor, cursor + duration, 1.0))
        cursor += duration
    return words


def test_gap_boundaries_for_pairing() -> None:
    attempt = raw(1, 0.0, "本タブレット表示にすると")  # ends at 1.8
    following_text = "本タブレット表示にすると、こうなります。"

    at_limit = [attempt, raw(2, 5.8, following_text)]
    assert len(token_restart_regions(at_limit)) == 1

    beyond = [attempt, raw(2, 5.81, following_text)]
    assert token_restart_regions(beyond) == []

    overlapping = [attempt, raw(2, 1.3, following_text)]
    assert len(token_restart_regions(overlapping)) == 1

    too_much_overlap = [attempt, raw(2, 1.29, following_text)]
    assert token_restart_regions(too_much_overlap) == []


def test_attempt_duration_limit() -> None:
    attempt = raw(1, 0.0, "本タブレット表示にすると", step=0.6)  # 7.2 seconds
    following = raw(2, 7.5, "本タブレット表示にすると、こうなります。", step=0.6)
    assert token_restart_regions([attempt, following]) == []

    regions = token_restart_regions([attempt, following], max_attempt_duration=10.0)
    assert len(regions) == 1


def test_short_full_respeak_and_complete_length_rules() -> None:
    # A 3-char incomplete attempt is flagged when fully re-spoken.
    segments = [raw(1, 0.0, "あめが"), raw(2, 0.6, "あめがふります")]
    assert len(token_restart_regions(segments)) == 1

    # A complete 5-char attempt is not flagged by a full re-speak...
    segments = [raw(1, 0.0, "あめがふる。"), raw(2, 1.0, "あめがふるでしょう")]
    assert token_restart_regions(segments) == []

    # ...but a complete attempt of 6+ chars is.
    segments = [raw(1, 0.0, "あめがふります。"), raw(2, 1.2, "あめがふりますね")]
    assert len(token_restart_regions(segments)) == 1


def test_partial_respeak_coverage_and_prefix_boundaries() -> None:
    attempt = raw(1, 0.0, "あいうえおかきくけこ")
    six_shared = raw(2, 1.7, "あいうえおかさしすせそ")
    regions = token_restart_regions([attempt, six_shared])
    assert len(regions) == 1
    assert regions[0].text == "あいうえおか"

    five_shared = raw(2, 1.7, "あいうえおさしすせそた")
    assert token_restart_regions([attempt, five_shared]) == []
    regions = token_restart_regions([attempt, five_shared], min_coverage=0.5)
    assert len(regions) == 1

    # Prefix below min_prefix never counts as a partial repeat.
    attempt = raw(1, 0.0, "あいうえお")
    three_shared = raw(2, 1.0, "あいうさしす")
    assert token_restart_regions([attempt, three_shared]) == []


def test_pair_region_text_is_truncated_to_14_chars() -> None:
    attempt = raw(1, 0.0, "あいうえおかきくけこさしすせそたちつてと")
    following = raw(2, 3.2, "あいうえおかきくけこさしすせそ、ですね")
    regions = token_restart_regions([attempt, following])
    assert len(regions) == 1
    assert regions[0].text == "あいうえおかきくけこさしすせ"


def test_exclusions_are_never_flagged() -> None:
    segments = [raw(1, 0.0, "なるほど"), raw(2, 0.8, "なるほど、次です")]
    assert token_restart_regions(segments) == []

    segments = [
        raw(1, 0.0, "はいはいはいはい"),
        raw(2, 1.4, "はいはいはいはい、次です"),
    ]
    assert token_restart_regions(segments) == []


def test_effective_end_discounts_outlier_last_token() -> None:
    words = words_from_durations("あいう", 0.0, [0.15, 0.15, 2.0])
    attempt = RawSegment(1, 0.0, 2.3, "あいう", 0.0, words)
    following = raw(2, 2.3, "あいうえお")
    regions = token_restart_regions([attempt, following])
    assert len(regions) == 1
    assert abs(regions[0].end - 0.45) < 1e-9


def test_effective_end_outlier_boundaries() -> None:
    # Last token of exactly 0.8s is not an outlier.
    words = words_from_durations("あいう", 0.0, [0.15, 0.15, 0.8])
    attempt = RawSegment(1, 0.0, 1.1, "あいう", 0.0, words)
    following = raw(2, 1.1, "あいうえお")
    regions = token_restart_regions([attempt, following])
    assert len(regions) == 1
    assert abs(regions[0].end - 1.1) < 1e-9

    # Last token exactly 2.5x the median is not an outlier either.
    words = words_from_durations("あいう", 0.0, [0.4, 0.4, 1.0])
    attempt = RawSegment(1, 0.0, 1.8, "あいう", 0.0, words)
    following = raw(2, 1.8, "あいうえお")
    regions = token_restart_regions([attempt, following])
    assert len(regions) == 1
    assert abs(regions[0].end - 1.8) < 1e-9

    # Just above the ratio, the median (upper middle) is used.
    words = words_from_durations("あいうえお", 0.0, [0.1, 0.2, 0.3, 0.4, 2.0])
    attempt = RawSegment(1, 0.0, 3.0, "あいうえお", 0.0, words)
    following = raw(2, 3.0, "あいうえおか")
    regions = token_restart_regions([attempt, following])
    assert len(regions) == 1
    assert abs(regions[0].end - 1.3) < 1e-9


def test_utterance_split_on_gap_boundary() -> None:
    # A gap of exactly pause_gap does not split: the within-utterance
    # repeat rule fires instead of the pair rule.
    words = char_words("あめがふる", 0.0)
    words += char_words("あめがふるでしょう", 1.15)
    segment = RawSegment(
        1, 0.0, words[-1].end, "あめがふるあめがふるでしょう", 0.0, words
    )
    regions = token_restart_regions([segment])
    assert len(regions) == 1
    assert regions[0].text == "あめがふる の繰り返し"

    # A slightly larger gap splits the utterance and the pair rule fires.
    words = char_words("あめがふる", 0.0)
    words += char_words("あめがふるでしょう", 1.16)
    segment = RawSegment(
        1, 0.0, words[-1].end, "あめがふるあめがふるでしょう", 0.0, words
    )
    regions = token_restart_regions([segment])
    assert len(regions) == 1
    assert regions[0].text == "あめがふる"


def test_utterance_split_on_long_token_boundary() -> None:
    # A previous token held for 1.0s splits the utterance: pair rule.
    words = words_from_durations("あめがふる", 0.0, [0.25, 0.25, 0.25, 0.25, 1.0])
    words += char_words("あめがふるでしょう", 2.0)
    segment = RawSegment(
        1, 0.0, words[-1].end, "あめがふるあめがふるでしょう", 0.0, words
    )
    regions = token_restart_regions([segment])
    assert len(regions) == 1
    assert regions[0].text == "あめがふる"

    # A 0.75s hold does not split: the within-utterance repeat rule fires.
    words = words_from_durations("あめがふる", 0.0, [0.25, 0.25, 0.25, 0.25, 0.75])
    words += char_words("あめがふるでしょう", 1.75)
    segment = RawSegment(
        1, 0.0, words[-1].end, "あめがふるあめがふるでしょう", 0.0, words
    )
    regions = token_restart_regions([segment])
    assert len(regions) == 1
    assert regions[0].text == "あめがふる の繰り返し"


def test_sentence_final_marks_attempt_complete_within_segment() -> None:
    words = char_words("あめがふる。あめがふるでしょう", 0.0)
    segment = RawSegment(
        1, 0.0, words[-1].end, "あめがふる。あめがふるでしょう", 0.0, words
    )
    assert token_restart_regions([segment]) == []


def test_punctuation_is_stripped_from_attempt_text() -> None:
    segments = [raw(1, 0.0, "あめが、ふる"), raw(2, 1.0, "あめがふるでしょう")]
    regions = token_restart_regions(segments)
    assert len(regions) == 1
    assert regions[0].text == "あめがふる"


def test_repeat_region_edge_cases() -> None:
    # A repetition that makes up the whole utterance is emphasis.
    assert token_restart_regions([raw(1, 0.0, "この辺はこの辺は")]) == []

    # A unit of one repeated character is not a phrase repeat.
    assert token_restart_regions([raw(1, 0.0, "ああああああああと")]) == []

    # With multi-character tokens the region spans whole tokens.
    words = [WordTimestamp("あしたのあしたの次です", 0.0, 0.7, 1.0)]
    segment = RawSegment(1, 0.0, 0.7, "あしたのあしたの次です", 0.0, words)
    regions = token_restart_regions([segment])
    assert len(regions) == 1
    assert regions[0].start == 0.0
    assert regions[0].end == 0.7
    assert regions[0].text == "あしたの の繰り返し"


def test_overlapping_regions_keep_the_first() -> None:
    # The pair region for the first utterance (0.0-1.65) overlaps its own
    # repeat region (0.0-0.6) by more than half of the shorter one, so it
    # is dropped; the second segment's own repeat region is unaffected.
    segments = [
        raw(1, 0.0, "あしたのあしたの次です"),
        raw(2, 1.8, "あしたのあしたの次ですね"),
    ]
    regions = token_restart_regions(segments)
    assert [region.text for region in regions] == [
        "あしたの の繰り返し",
        "あしたの の繰り返し",
    ]
    assert [region.start for region in regions] == [0.0, 1.8]
