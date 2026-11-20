import pytest

from podcastpreprocessor.models import (
    AudioSlice,
    AudioSource,
    ClipMarker,
    ProcessedSegment,
    RawSegment,
    RefinedSegment,
    WordTimestamp,
    seconds_to_frame,
    seconds_to_frame_range,
)


def test_seconds_to_frame_rounds_to_nearest_frame() -> None:
    assert seconds_to_frame(1.0, 30) == 30
    assert seconds_to_frame(1.02, 30) == 31


def test_seconds_to_frame_range_rejects_ranges_lost_to_rounding() -> None:
    assert seconds_to_frame_range(1.001, 1.01, 30) is None
    assert seconds_to_frame_range(1.0, 1.02, 30) == (30, 31)
    assert seconds_to_frame_range(9.9, 10.1, 30, max_end_frame=300) == (297, 300)


def test_models_reject_invalid_ranges() -> None:
    WordTimestamp("zero duration", 1.0, 1.0, 0.9)

    with pytest.raises(ValueError, match="word timestamp must satisfy"):
        WordTimestamp("reversed", 2.0, 1.0, 0.9)

    with pytest.raises(ValueError, match="segment must satisfy"):
        RawSegment(1, 2.0, 1.0, "bad", 0.0)

    with pytest.raises(ValueError, match="segment must satisfy"):
        RawSegment(1, float("nan"), 1.0, "bad", 0.0)

    with pytest.raises(ValueError, match="timeline frames must satisfy"):
        ProcessedSegment("s1", 1, 30, 30, 30, 31, "text")


def test_seconds_to_frame_rejects_invalid_inputs() -> None:
    with pytest.raises(
        ValueError, match="seconds must be a finite non-negative number"
    ):
        seconds_to_frame(-0.1, 30)
    with pytest.raises(
        ValueError, match="seconds must be a finite non-negative number"
    ):
        seconds_to_frame(float("nan"), 30)
    with pytest.raises(
        ValueError, match="seconds must be a finite non-negative number"
    ):
        seconds_to_frame(float("inf"), 30)
    with pytest.raises(ValueError, match="fps must be positive"):
        seconds_to_frame(1.0, 0)
    with pytest.raises(ValueError, match="fps must be positive"):
        seconds_to_frame(1.0, -30)


def test_seconds_to_frame_rounding_boundaries() -> None:
    assert seconds_to_frame(0.0, 30) == 0
    assert seconds_to_frame(0.5, 1) == 0
    assert seconds_to_frame(1.5, 1) == 2
    assert seconds_to_frame(2.5, 1) == 2
    assert seconds_to_frame(0.25, 30) == 8
    assert seconds_to_frame(0.15, 30) == 4


def test_seconds_to_frame_range_validates_inputs() -> None:
    with pytest.raises(ValueError, match="range endpoints must be finite"):
        seconds_to_frame_range(float("nan"), 1.0, 30)
    with pytest.raises(ValueError, match="range endpoints must be finite"):
        seconds_to_frame_range(0.0, float("inf"), 30)
    with pytest.raises(ValueError, match="range must satisfy 0 <= start < end"):
        seconds_to_frame_range(-0.5, 1.0, 30)
    with pytest.raises(ValueError, match="range must satisfy 0 <= start < end"):
        seconds_to_frame_range(1.0, 1.0, 30)
    with pytest.raises(ValueError, match="range must satisfy 0 <= start < end"):
        seconds_to_frame_range(2.0, 1.0, 30)
    with pytest.raises(ValueError, match="max_end_frame must be positive"):
        seconds_to_frame_range(0.0, 1.0, 30, max_end_frame=0)
    with pytest.raises(ValueError, match="max_end_frame must be positive"):
        seconds_to_frame_range(0.0, 1.0, 30, max_end_frame=-10)
    with pytest.raises(ValueError, match="fps must be positive"):
        seconds_to_frame_range(0.0, 1.0, 0)


def test_seconds_to_frame_range_clamps_start_too() -> None:
    assert seconds_to_frame_range(10.0, 10.1, 30, max_end_frame=300) is None
    assert seconds_to_frame_range(0.0, 0.04, 30) == (0, 1)
    assert seconds_to_frame_range(0.0, 1.0, 30, max_end_frame=10) == (0, 10)


def test_word_timestamp_boundaries() -> None:
    WordTimestamp("zero", 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="word timestamp must satisfy"):
        WordTimestamp("negative", -0.1, 1.0, 0.9)
    with pytest.raises(ValueError, match="word timestamp must satisfy"):
        WordTimestamp("nan end", 0.0, float("nan"), 0.9)
    with pytest.raises(ValueError, match="word timestamp must satisfy"):
        WordTimestamp("infinite", float("inf"), float("inf"), 0.9)


def test_raw_segment_boundaries() -> None:
    RawSegment(1, 0.0, 0.1, "ok", 0.0)
    RawSegment(1, 0.0, 0.1, "ok", 1.0)

    with pytest.raises(ValueError, match="segment must satisfy"):
        RawSegment(1, 1.0, 1.0, "empty", 0.0)
    with pytest.raises(ValueError, match="segment must satisfy"):
        RawSegment(1, -0.1, 1.0, "negative", 0.0)
    with pytest.raises(ValueError, match="segment must satisfy"):
        RawSegment(1, 0.0, float("inf"), "infinite", 0.0)
    with pytest.raises(ValueError, match="no_speech_prob must be between 0 and 1"):
        RawSegment(1, 0.0, 1.0, "bad prob", -0.01)
    with pytest.raises(ValueError, match="no_speech_prob must be between 0 and 1"):
        RawSegment(1, 0.0, 1.0, "bad prob", 1.01)


def test_refined_segment_validation() -> None:
    RefinedSegment("s1", 1, 0.0, 0.5, "text", 0.5)

    with pytest.raises(ValueError, match="segment must satisfy"):
        RefinedSegment("s1", 1, 0.5, 0.5, "text", 0.5)
    with pytest.raises(ValueError, match="segment must satisfy"):
        RefinedSegment("s1", 1, -0.1, 0.5, "text", 0.5)
    with pytest.raises(ValueError, match="segment must satisfy"):
        RefinedSegment("s1", 1, float("nan"), 0.5, "text", 0.5)
    with pytest.raises(ValueError, match="track_index must be positive"):
        RefinedSegment("s1", 0, 0.0, 0.5, "text", 0.5)
    with pytest.raises(ValueError, match="track_index must be positive"):
        RefinedSegment("s1", -2, 0.0, 0.5, "text", 0.5)


def test_clip_marker_validation() -> None:
    ClipMarker("m", 0, 1)

    with pytest.raises(ValueError, match="clip marker must satisfy"):
        ClipMarker("m", -1, 1)
    with pytest.raises(ValueError, match="clip marker must satisfy"):
        ClipMarker("m", 5, 5)
    with pytest.raises(ValueError, match="clip marker must satisfy"):
        ClipMarker("m", 5, 4)


def test_processed_segment_validation() -> None:
    ProcessedSegment("s1", 1, 0, 30, 0, 30, "text")

    with pytest.raises(ValueError, match="track_index must be positive"):
        ProcessedSegment("s1", 0, 0, 30, 0, 30, "text")
    with pytest.raises(ValueError, match="timeline frames must satisfy"):
        ProcessedSegment("s1", 1, -1, 30, 0, 30, "text")
    with pytest.raises(ValueError, match="timeline frames must satisfy"):
        ProcessedSegment("s1", 1, 30, 29, 0, 30, "text")
    with pytest.raises(ValueError, match="source frames must satisfy"):
        ProcessedSegment("s1", 1, 0, 30, -1, 30, "text")
    with pytest.raises(ValueError, match="source frames must satisfy"):
        ProcessedSegment("s1", 1, 0, 30, 30, 30, "text")
    with pytest.raises(ValueError, match="source frames must satisfy"):
        ProcessedSegment("s1", 1, 0, 30, 30, 29, "text")
    with pytest.raises(ValueError, match="source_id must not be empty"):
        ProcessedSegment("s1", 1, 0, 30, 0, 30, "text", source_id="")


def test_audio_source_validation() -> None:
    AudioSource("src", "/a.wav", 1, 100)

    with pytest.raises(ValueError, match="source_id must not be empty"):
        AudioSource("", "/a.wav", 1, 100)
    with pytest.raises(ValueError, match="path must not be empty"):
        AudioSource("src", "", 1, 100)
    with pytest.raises(ValueError, match="track_index must be positive"):
        AudioSource("src", "/a.wav", 0, 100)
    with pytest.raises(ValueError, match="duration_frames must be positive"):
        AudioSource("src", "/a.wav", 1, 0)
    with pytest.raises(ValueError, match="duration_frames must be positive"):
        AudioSource("src", "/a.wav", 1, -5)


def test_audio_slice_validation() -> None:
    AudioSlice("src", 0, 30, 10, 40)

    with pytest.raises(ValueError, match="timeline slice must have a positive range"):
        AudioSlice("src", -1, 30, 10, 40)
    with pytest.raises(ValueError, match="timeline slice must have a positive range"):
        AudioSlice("src", 30, 30, 10, 40)
    with pytest.raises(ValueError, match="source slice must have a positive range"):
        AudioSlice("src", 0, 30, -1, 29)
    with pytest.raises(ValueError, match="source slice must have a positive range"):
        AudioSlice("src", 0, 30, 10, 10)
    with pytest.raises(
        ValueError, match="timeline and source slice durations must match"
    ):
        AudioSlice("src", 0, 30, 10, 41)
