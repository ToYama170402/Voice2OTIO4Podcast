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
