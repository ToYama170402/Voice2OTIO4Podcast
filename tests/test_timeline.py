import pytest

from podcastpreprocessor.models import ProcessedSegment
from podcastpreprocessor.timeline import build_audio_slices


def test_audio_slices_include_leading_middle_and_trailing_gaps() -> None:
    segments = [
        ProcessedSegment("s1", 1, 30, 60, 30, 60, "one", source_id="speaker_a"),
        ProcessedSegment("s2", 1, 90, 120, 90, 120, "two", source_id="speaker_a"),
    ]

    slices = build_audio_slices(segments, "speaker_a", duration_frames=150)

    assert [(item.start_frame, item.end_frame) for item in slices] == [
        (0, 30),
        (30, 60),
        (60, 90),
        (90, 120),
        (120, 150),
    ]
    assert sum(item.end_frame - item.start_frame for item in slices) == 150


def test_audio_slices_reject_overlapping_segments() -> None:
    segments = [
        ProcessedSegment("s1", 1, 0, 60, 0, 60, "one", source_id="speaker_a"),
        ProcessedSegment("s2", 1, 30, 90, 30, 90, "two", source_id="speaker_a"),
    ]

    with pytest.raises(ValueError, match="overlap"):
        build_audio_slices(segments, "speaker_a")


def test_audio_slices_reject_non_positive_duration() -> None:
    segment = ProcessedSegment("s1", 1, 0, 30, 0, 30, "one")

    with pytest.raises(ValueError, match="duration_frames"):
        build_audio_slices([segment], "source_1", duration_frames=0)
