import pytest

from podcastpreprocessor.evaluator import (
    evaluate_segment,
    evaluate_text_pattern,
    filler_core,
    is_filler_text,
)
from podcastpreprocessor.models import RefinedSegment


def segment(
    text: str, start: float = 0.0, end: float = 1.0, probability: float = 0.0
) -> RefinedSegment:
    return RefinedSegment("s1", 1, start, end, text, probability)


def test_evaluator_detects_pause_filler_and_non_speech() -> None:
    result = evaluate_segment(segment("えーと", 2.0, 3.0, 0.81), previous_end=1.0)

    assert result.is_cut_candidate
    assert result.reasons == ("無音 1.0秒", "フィラー (えーと)", "非音声確率 0.81")
    assert result.flag_label.startswith("[要カット:")


def test_pause_threshold_is_strictly_greater_than_point_four() -> None:
    result = evaluate_segment(segment("通常", 1.4, 2.0), previous_end=1.0)

    assert not result.is_cut_candidate


def test_evaluator_detects_standalone_filler_variants() -> None:
    # あの is not in the substring filler list, but a segment that is only
    # a filler utterance (as produced by filler isolation) must be flagged.
    result = evaluate_segment(segment("あの、"))

    assert result.is_cut_candidate
    assert result.reasons == ("フィラー (あの)",)


def test_evaluator_does_not_flag_ordinary_speech() -> None:
    result = evaluate_segment(segment("その話をしました"))

    assert not result.is_cut_candidate


def test_evaluator_supports_custom_regex() -> None:
    result = evaluate_text_pattern(segment("そのー話", end=2.0), r"そのー")

    assert result.is_cut_candidate
    assert "フィラー (そのー話)" in result.marker_reason


def test_evaluation_label_formats() -> None:
    empty = evaluate_segment(segment("通常の文"))
    assert empty.flag_label == ""
    assert empty.marker_reason == ""

    result = evaluate_segment(segment("えーと", 2.0, 3.0, 0.81), previous_end=1.0)
    assert (
        result.flag_label
        == "[要カット: 無音 1.0秒 / フィラー (えーと) / 非音声確率 0.81]"
    )
    assert (
        result.marker_reason == "判定: 無音 1.0秒 / フィラー (えーと) / 非音声確率 0.81"
    )


def test_no_speech_threshold_is_strict() -> None:
    assert not evaluate_segment(segment("通常", probability=0.8)).is_cut_candidate
    result = evaluate_segment(segment("通常", probability=0.5), no_speech_threshold=0.5)
    assert not result.is_cut_candidate
    result = evaluate_segment(
        segment("通常", probability=0.51), no_speech_threshold=0.5
    )
    assert result.reasons == ("非音声確率 0.51",)


def test_evaluator_validates_thresholds() -> None:
    with pytest.raises(ValueError, match="pause_threshold must be non-negative"):
        evaluate_segment(segment("通常"), pause_threshold=-0.1)
    with pytest.raises(ValueError, match="no_speech_threshold must be between 0 and 1"):
        evaluate_segment(segment("通常"), no_speech_threshold=-0.1)
    with pytest.raises(ValueError, match="no_speech_threshold must be between 0 and 1"):
        evaluate_segment(segment("通常"), no_speech_threshold=1.1)

    evaluate_segment(segment("通常"), no_speech_threshold=0.0)
    evaluate_segment(segment("通常"), no_speech_threshold=1.0)
    result = evaluate_segment(
        segment("通常", 1.1, 2.0), previous_end=1.0, pause_threshold=0.0
    )
    assert result.reasons == ("無音 0.1秒",)


def test_pause_requires_previous_end() -> None:
    result = evaluate_segment(segment("通常", 10.0, 11.0), previous_end=None)
    assert not result.is_cut_candidate

    result = evaluate_segment(segment("通常", 2.3, 3.0), previous_end=1.0)
    assert result.reasons == ("無音 1.3秒",)


def test_multiple_fillers_follow_filler_list_order() -> None:
    result = evaluate_segment(segment("まー、えーと"))
    assert result.reasons == ("フィラー (えーと、まー)",)

    result = evaluate_segment(segment("今日はえーとですね"))
    assert result.reasons == ("フィラー (えーと)",)


def test_filler_core_and_pattern_variants() -> None:
    assert filler_core(" えーと。、 ") == "えーと"
    assert filler_core("あ の") == "あの"
    assert filler_core("…えっと！") == "えっと"

    assert is_filler_text("えーーと")
    assert is_filler_text("まあ")
    assert not is_filler_text("ま")
    assert not is_filler_text("えーと話")
    assert not is_filler_text("")

    result = evaluate_segment(segment("うーん"))
    assert result.reasons == ("フィラー (うーん)",)
    # えっ is a substring filler, so it wins over the whole-utterance pattern.
    result = evaluate_segment(segment(" えっと "))
    assert result.reasons == ("フィラー (えっ)",)


def test_text_pattern_appends_reason_after_existing_ones() -> None:
    result = evaluate_text_pattern(segment("えーと"), r"えー")
    assert result.reasons == ("フィラー (えーと)", "フィラー (えーと)")

    result = evaluate_text_pattern(segment("通常の文"), r"そのー")
    assert not result.is_cut_candidate
    assert result.reasons == ()

    with pytest.raises(ValueError, match="pause_threshold must be non-negative"):
        evaluate_text_pattern(segment("通常"), r"x", pause_threshold=-1.0)
