import pytest

from podcastpreprocessor.models import (
    ProcessedSegment,
    RawSegment,
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

    with pytest.raises(ValueError):
        WordTimestamp("reversed", 2.0, 1.0, 0.9)

    with pytest.raises(ValueError):
        RawSegment(1, 2.0, 1.0, "bad", 0.0)

    with pytest.raises(ValueError):
        RawSegment(1, float("nan"), 1.0, "bad", 0.0)

    with pytest.raises(ValueError):
        ProcessedSegment("s1", 1, 30, 30, 30, 31, "text")
