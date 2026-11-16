from podcastpreprocessor.evaluator import detect_restarts
from podcastpreprocessor.models import RawSegment, RefinedSegment
from podcastpreprocessor.pipeline import process_segments


def refined(segment_id: str, start: float, end: float, text: str) -> RefinedSegment:
    return RefinedSegment(segment_id, 1, start, end, text, 0.0)


def test_full_respeak_is_flagged() -> None:
    segments = [
        refined("a", 0.0, 2.0, "本タブレット表示にすると"),
        refined("b", 2.3, 5.0, "本タブレット表示にすると、こうなります。"),
    ]
    reasons = detect_restarts(segments)
    assert reasons == {"a": "言い直し (本タブレット表示にすると)"}


def test_partial_respeak_of_incomplete_attempt_is_flagged() -> None:
    segments = [
        refined("a", 0.0, 2.0, "野菜とか乗っけててる"),
        refined("b", 2.2, 5.0, "野菜とか乗っけて焼くと美味しいです。"),
    ]
    reasons = detect_restarts(segments)
    assert list(reasons) == ["a"]
    assert reasons["a"].startswith("言い直し (野菜とか乗っけて")


def test_short_incomplete_full_respeak_is_flagged() -> None:
    segments = [
        refined("a", 0.0, 1.0, "次は駅"),
        refined("b", 1.2, 3.0, "次は駅の話です。"),
    ]
    assert list(detect_restarts(segments)) == ["a"]


def test_complete_sentence_with_shared_opening_is_not_flagged() -> None:
    segments = [
        refined("a", 0.0, 2.5, "私たちは昨日行きました。"),
        refined("b", 2.8, 5.0, "私たちは今日行きます。"),
    ]
    assert detect_restarts(segments) == {}


def test_backchannel_is_not_flagged() -> None:
    segments = [
        refined("a", 0.0, 0.8, "そうだね。"),
        refined("b", 1.0, 3.0, "そうだね、じゃあ次に進もう。"),
    ]
    assert detect_restarts(segments) == {}


def test_filler_segment_is_not_flagged_as_restart() -> None:
    segments = [
        refined("a", 0.0, 0.8, "えーと"),
        refined("b", 1.0, 3.0, "えーと、次は駅の話です。"),
    ]
    assert detect_restarts(segments) == {}


def test_large_gap_is_not_flagged() -> None:
    segments = [
        refined("a", 0.0, 2.0, "本タブレット表示にすると"),
        refined("b", 12.0, 15.0, "本タブレット表示にすると、こうなります。"),
    ]
    assert detect_restarts(segments) == {}


def test_long_attempt_is_not_flagged() -> None:
    segments = [
        refined("a", 0.0, 8.0, "本タブレット表示にすると"),
        refined("b", 8.3, 11.0, "本タブレット表示にすると、こうなります。"),
    ]
    assert detect_restarts(segments) == {}


def test_within_segment_repeat_is_flagged() -> None:
    segments = [
        refined("a", 0.0, 2.5, "どのぐらいだっけ？どのぐらいだっけ？話したの？"),
        refined("b", 3.0, 5.0, "二週間目か。"),
    ]
    reasons = detect_restarts(segments)
    assert reasons == {"a": "言い直し (どのぐらいだっけ の繰り返し)"}


def test_ordinary_segments_are_not_flagged() -> None:
    segments = [
        refined("a", 0.0, 2.0, "今日はいい天気です。"),
        refined("b", 2.3, 4.5, "明日も晴れるといいですね。"),
    ]
    assert detect_restarts(segments) == {}


def test_process_segments_marks_restart_as_cut_candidate(tmp_path) -> None:
    raw = [
        RawSegment(1, 0.0, 2.0, "本タブレット表示にすると", 0.0),
        RawSegment(2, 2.3, 5.0, "本タブレット表示にすると、こうなります。", 0.0),
        RawSegment(3, 5.5, 7.0, "次の話題です。", 0.0),
    ]
    processed = process_segments(raw, "C:/audio/a.wav", str(tmp_path / "out.otio"))
    by_text = {segment.subtitle_text: segment for segment in processed}
    attempt = by_text["本タブレット表示にすると"]
    assert attempt.is_cut_candidate
    assert "言い直し" in attempt.marker_reason
    assert not by_text["本タブレット表示にすると、こうなります。"].is_cut_candidate


def test_emphatic_repetition_is_not_flagged() -> None:
    segments = [
        refined("a", 0.0, 1.0, "そうそうそうそう。"),
        refined("b", 1.2, 3.0, "なんかあれ、東日本という西日本で違ったよ"),
        refined("c", 3.5, 4.5, "してないしてない。"),
        refined("d", 4.8, 6.0, "うん。"),
    ]
    assert detect_restarts(segments) == {}


def test_repeat_with_continuation_is_flagged() -> None:
    segments = [
        refined("a", 0.0, 3.6, "この辺はこの辺は多分スマホだと"),
        refined("b", 3.8, 4.5, "思う。"),
    ]
    reasons = detect_restarts(segments)
    assert reasons == {"a": "言い直し (この辺は の繰り返し)"}


def test_short_complete_respeak_is_not_flagged() -> None:
    segments = [
        refined("a", 0.0, 0.8, "ちょっと。"),
        refined("b", 1.0, 3.0, "ちょっと長い話なんですけど。"),
    ]
    assert detect_restarts(segments) == {}
