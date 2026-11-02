import pytest

from podcastpreprocessor.models import (
    ProcessedSegment,
    RawSegment,
    WordTimestamp,
    seconds_to_frame,
)


def test_seconds_to_frame_rounds_to_nearest_frame() -> None:
    assert seconds_to_frame(1.0, 30) == 30
    assert seconds_to_frame(1.02, 30) == 31


def test_models_reject_invalid_ranges() -> None:
    with pytest.raises(ValueError):
        WordTimestamp("bad", 1.0, 1.0, 0.9)

    with pytest.raises(ValueError):
        RawSegment(1, 2.0, 1.0, "bad", 0.0)

    with pytest.raises(ValueError):
        ProcessedSegment("s1", 1, 30, 30, 30, 31, "text")