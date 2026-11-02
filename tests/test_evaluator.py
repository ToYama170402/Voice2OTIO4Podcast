from podcastpreprocessor.evaluator import evaluate_segment, evaluate_text_pattern
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


def test_evaluator_supports_custom_regex() -> None:
    result = evaluate_text_pattern(segment("そのー話", end=2.0), r"そのー")

    assert result.is_cut_candidate
    assert "フィラー (そのー話)" in result.marker_reason
