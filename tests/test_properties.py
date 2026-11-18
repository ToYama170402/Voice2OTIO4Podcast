"""Property-based tests: invariants that must hold for any input.

The unit tests pin down specific examples; these tests state the
contracts the rest of the pipeline relies on (ordering, bounds,
round-trips) and let Hypothesis search for counterexamples.
"""

import re
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, settings
from hypothesis import strategies as st

from podcastpreprocessor.filler_detector import token_filler_regions
from podcastpreprocessor.models import (
    ProcessedSegment,
    RawSegment,
    WordTimestamp,
    seconds_to_frame,
    seconds_to_frame_range,
)
from podcastpreprocessor.restart_detector import token_restart_regions
from podcastpreprocessor.segmentator import segment_source
from podcastpreprocessor.srt import write_srt
from podcastpreprocessor.transcript_cache import (
    ChunkCache,
    _segment_from_dict,
    _segment_to_dict,
)

settings.register_profile(
    "ci", max_examples=40, deadline=timedelta(seconds=5), derandomize=False
)
settings.load_profile("ci")

_TOKEN_TEXTS = [
    "えー",
    "あの",
    "その",
    "この",
    "今日",
    "は",
    "テスト",
    "です",
    "ます",
    "昨日",
    "明日",
    "、",
    "。",
]


@st.composite
def raw_segment_lists(draw: st.DrawFn) -> list[RawSegment]:
    """Random segments whose word tokens tile the segment text."""
    segments: list[RawSegment] = []
    cursor = draw(st.floats(min_value=0.0, max_value=5.0))
    for index in range(draw(st.integers(min_value=1, max_value=4))):
        words: list[WordTimestamp] = []
        for _ in range(draw(st.integers(min_value=1, max_value=12))):
            text = draw(st.sampled_from(_TOKEN_TEXTS))
            duration = draw(st.floats(min_value=0.05, max_value=0.6))
            words.append(WordTimestamp(text, cursor, cursor + duration, 0.9))
            cursor += duration + draw(st.sampled_from([0.0, 0.05, 0.2, 0.9]))
        segments.append(
            RawSegment(
                id=index,
                start=words[0].start,
                end=words[-1].end,
                text="".join(w.word for w in words),
                no_speech_prob=0.1,
                words=words,
                source_id="source_1",
            )
        )
        cursor += draw(st.floats(min_value=0.0, max_value=1.0))
    return segments


@given(
    seconds=st.floats(min_value=0.0, max_value=36000.0),
    fps=st.integers(min_value=1, max_value=120),
)
def test_seconds_to_frame_roundtrip(seconds: float, fps: int) -> None:
    frame = seconds_to_frame(seconds, fps)
    assert frame >= 0
    assert abs(frame / fps - seconds) <= 0.5 / fps + 1e-9


@given(
    start=st.floats(min_value=0.0, max_value=1000.0),
    length=st.floats(min_value=0.001, max_value=100.0),
    fps=st.integers(min_value=1, max_value=120),
)
def test_frame_range_is_valid(start: float, length: float, fps: int) -> None:
    result = seconds_to_frame_range(start, start + length, fps)
    if result is not None:
        start_frame, end_frame = result
        assert 0 <= start_frame < end_frame
        assert start_frame == seconds_to_frame(start, fps)


@given(segments=raw_segment_lists())
def test_segment_roundtrip_through_cache_dict(segments: list[RawSegment]) -> None:
    for segment in segments:
        restored = _segment_from_dict(_segment_to_dict(segment), "source_1")
        assert restored == segment


@given(segments=raw_segment_lists())
def test_chunk_cache_roundtrip(segments: list[RawSegment]) -> None:
    # A fresh directory per execution: Hypothesis re-runs the body with
    # related inputs while shrinking, so shared paths would leak state.
    with TemporaryDirectory() as directory:
        cache = ChunkCache(Path(directory) / "partials" / "key.jsonl")
        for index, segment in enumerate(segments):
            cache.store(index, segment)
        cache.store(len(segments), None)

        entries = cache.load()

    assert len(entries) == len(segments) + 1
    for index, segment in enumerate(segments):
        assert entries[index] == segment
    assert entries[len(segments)] is None


@given(segments=raw_segment_lists())
def test_restart_regions_are_ordered_and_in_bounds(
    segments: list[RawSegment],
) -> None:
    regions = token_restart_regions(segments)
    first_start = min(w.start for s in segments for w in s.words)
    last_end = max(w.end for s in segments for w in s.words)

    previous_end = first_start
    for region in regions:
        assert first_start <= region.start < region.end <= last_end
        assert region.start >= previous_end
        previous_end = region.end


@given(segments=raw_segment_lists())
def test_filler_regions_align_with_word_bounds(segments: list[RawSegment]) -> None:
    regions = token_filler_regions(segments)
    starts = {w.start for s in segments for w in s.words}
    ends = {w.end for s in segments for w in s.words}

    previous_end = 0.0
    for region in regions:
        assert region.start in starts
        assert region.end in ends
        assert region.start < region.end
        assert region.start >= previous_end
        previous_end = region.end


@given(segments=raw_segment_lists())
def test_segmentation_pieces_are_ordered_and_in_bounds(
    segments: list[RawSegment],
) -> None:
    pieces = segment_source(segments, track_index=1)
    first_start = min(s.start for s in segments)
    last_end = max(s.end for s in segments)

    assert pieces
    previous_end = first_start
    for piece in pieces:
        assert first_start <= piece.start < piece.end <= last_end
        assert piece.start >= previous_end
        assert piece.text
        previous_end = piece.end


@st.composite
def processed_segment_lists(draw: st.DrawFn) -> list[ProcessedSegment]:
    """Random valid processed segments with increasing frame ranges."""
    pieces: list[ProcessedSegment] = []
    cursor = draw(st.integers(min_value=0, max_value=100))
    for index in range(draw(st.integers(min_value=1, max_value=6))):
        length = draw(st.integers(min_value=1, max_value=300))
        text = draw(st.sampled_from(_TOKEN_TEXTS))
        pieces.append(
            ProcessedSegment(
                segment_id=f"s{index}",
                track_index=1,
                start_frame=cursor,
                end_frame=cursor + length,
                in_frame=cursor,
                out_frame=cursor + length,
                subtitle_text=text,
            )
        )
        cursor += length + draw(st.integers(min_value=0, max_value=50))
    return pieces


@given(pieces=processed_segment_lists())
def test_srt_timestamps_are_well_formed(pieces: list[ProcessedSegment]) -> None:
    with TemporaryDirectory() as directory:
        output = Path(directory) / "out.srt"
        write_srt(str(output), pieces, fps=30)
        lines = output.read_text(encoding="utf-8").splitlines()

    timing_lines = [line for line in lines if " --> " in line]
    assert len(timing_lines) == len(pieces)
    pattern = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$")
    for line in timing_lines:
        assert pattern.match(line)
        start, end = line.split(" --> ")
        assert start < end
