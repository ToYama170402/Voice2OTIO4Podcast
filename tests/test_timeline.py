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


def test_audio_slices_without_segments_cover_the_whole_source() -> None:
    slices = build_audio_slices([], "speaker_a", duration_frames=900)

    assert len(slices) == 1
    item = slices[0]
    assert (item.start_frame, item.end_frame) == (0, 900)
    assert (item.source_in_frame, item.source_out_frame) == (0, 900)
    assert item.source_id == "speaker_a"
    assert item.segment_id is None

    # Segments of other sources do not count.
    other = ProcessedSegment("s1", 1, 0, 30, 0, 30, "one", source_id="speaker_b")
    slices = build_audio_slices([other], "speaker_a", duration_frames=60)
    assert [(item.start_frame, item.end_frame) for item in slices] == [(0, 60)]


def test_audio_slices_without_segments_require_a_duration() -> None:
    with pytest.raises(
        ValueError, match="duration_frames is required when there are no segments"
    ):
        build_audio_slices([], "speaker_a")
    with pytest.raises(
        ValueError, match="duration_frames is required when there are no segments"
    ):
        build_audio_slices([], "speaker_a", duration_frames=0)
    with pytest.raises(
        ValueError, match="duration_frames is required when there are no segments"
    ):
        build_audio_slices([], "speaker_a", duration_frames=-10)


def test_audio_slices_reject_duration_shorter_than_a_segment() -> None:
    segment = ProcessedSegment("s1", 1, 0, 90, 0, 90, "one", source_id="speaker_a")

    with pytest.raises(
        ValueError, match="duration_frames cannot be shorter than a segment"
    ):
        build_audio_slices([segment], "speaker_a", duration_frames=60)

    # A duration equal to the last segment's end is accepted.
    slices = build_audio_slices([segment], "speaker_a", duration_frames=90)
    assert [(item.start_frame, item.end_frame) for item in slices] == [(0, 90)]


def test_audio_slices_for_adjacent_segments_have_no_gap_slices() -> None:
    segments = [
        ProcessedSegment("s2", 1, 30, 60, 30, 60, "two", source_id="speaker_a"),
        ProcessedSegment("s1", 1, 0, 30, 0, 30, "one", source_id="speaker_a"),
    ]

    slices = build_audio_slices(segments, "speaker_a")

    assert [(item.start_frame, item.end_frame) for item in slices] == [
        (0, 30),
        (30, 60),
    ]
    assert [item.segment_id for item in slices] == ["s1", "s2"]
    # Without an explicit duration the source ends at the last segment.
    assert slices[-1].source_out_frame == 60


def test_audio_slices_link_segment_ids_and_source_frames() -> None:
    segment = ProcessedSegment("s1", 1, 30, 60, 300, 330, "one", source_id="speaker_a")

    slices = build_audio_slices([segment], "speaker_a", duration_frames=60)

    gap, body = slices
    assert gap.segment_id is None
    assert (gap.source_in_frame, gap.source_out_frame) == (0, 30)
    assert body.segment_id == "s1"
    assert (body.source_in_frame, body.source_out_frame) == (300, 330)
