"""Core data models shared by the processing stages."""

import math
from dataclasses import dataclass, field


def seconds_to_frame(seconds: float, fps: int) -> int:
    """Convert seconds to the nearest timeline frame."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be a finite non-negative number")
    if fps <= 0:
        raise ValueError("fps must be positive")
    return round(seconds * fps)


def seconds_to_frame_range(
    start: float,
    end: float,
    fps: int,
    max_end_frame: int | None = None,
) -> tuple[int, int] | None:
    """Convert a positive seconds range, or return None if it rounds away."""
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("range endpoints must be finite")
    if start < 0 or end <= start:
        raise ValueError("range must satisfy 0 <= start < end")
    if max_end_frame is not None and max_end_frame <= 0:
        raise ValueError("max_end_frame must be positive")

    start_frame = seconds_to_frame(start, fps)
    end_frame = seconds_to_frame(end, fps)
    if max_end_frame is not None:
        start_frame = min(start_frame, max_end_frame)
        end_frame = min(end_frame, max_end_frame)
    if end_frame <= start_frame:
        return None
    return start_frame, end_frame


@dataclass(frozen=True)
class WordTimestamp:
    word: str
    start: float
    end: float
    probability: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.start)
            or not math.isfinite(self.end)
            or self.start < 0
            or self.end < self.start
        ):
            raise ValueError("word timestamp must satisfy 0 <= start <= end")


@dataclass
class RawSegment:
    id: int
    start: float
    end: float
    text: str
    no_speech_prob: float
    words: list[WordTimestamp] = field(default_factory=list)
    source_id: str = "source_1"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.start)
            or not math.isfinite(self.end)
            or self.start < 0
            or self.end <= self.start
        ):
            raise ValueError("segment must satisfy 0 <= start < end")
        if not 0 <= self.no_speech_prob <= 1:
            raise ValueError("no_speech_prob must be between 0 and 1")


@dataclass
class RefinedSegment:
    segment_id: str
    track_index: int
    start: float
    end: float
    text: str
    no_speech_prob: float
    words: list[WordTimestamp] = field(default_factory=list)
    source_id: str = "source_1"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.start)
            or not math.isfinite(self.end)
            or self.start < 0
            or self.end <= self.start
        ):
            raise ValueError("segment must satisfy 0 <= start < end")
        if self.track_index <= 0:
            raise ValueError("track_index must be positive")


@dataclass(frozen=True)
class ClipMarker:
    """A marker attached to an audio clip, in frames relative to its start."""

    name: str
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError("clip marker must satisfy 0 <= start < end")


@dataclass
class ProcessedSegment:
    segment_id: str
    track_index: int
    start_frame: int
    end_frame: int
    in_frame: int
    out_frame: int
    subtitle_text: str
    is_cut_candidate: bool = False
    flag_label: str = ""
    marker_reason: str = ""
    color_label: str = "None"
    source_id: str = "source_1"
    audio_path: str = ""
    link_group_id: int = 0
    clip_markers: list[ClipMarker] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.track_index <= 0:
            raise ValueError("track_index must be positive")
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError("timeline frames must satisfy 0 <= start < end")
        if self.in_frame < 0 or self.out_frame <= self.in_frame:
            raise ValueError("source frames must satisfy 0 <= in < out")
        if not self.source_id:
            raise ValueError("source_id must not be empty")


@dataclass(frozen=True)
class AudioInput:
    """One transcription input: a file and one audio stream inside it.

    A file can hold several audio streams (tracks); each stream is
    processed as its own source. ``stream_index`` counts audio streams
    only, starting at 0.
    """

    path: str
    stream_index: int = 0

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("path must not be empty")
        if self.stream_index < 0:
            raise ValueError("stream_index must be non-negative")


def as_audio_input(value: AudioInput | str) -> AudioInput:
    """Coerce a plain path to an AudioInput for the first audio stream."""
    return value if isinstance(value, AudioInput) else AudioInput(str(value))


@dataclass(frozen=True)
class AudioStreamInfo:
    """Facts about one audio stream inside a media file."""

    stream_index: int
    channels: int
    sample_rate: int

    def __post_init__(self) -> None:
        if self.stream_index < 0:
            raise ValueError("stream_index must be non-negative")
        if self.channels < 0:
            raise ValueError("channels must be non-negative")
        if self.sample_rate < 0:
            raise ValueError("sample_rate must be non-negative")


@dataclass(frozen=True)
class AudioSource:
    source_id: str
    path: str
    track_index: int
    duration_frames: int
    name: str = ""
    stream_index: int = 0
    stream_count: int = 1
    channels: int = 0

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if not self.path:
            raise ValueError("path must not be empty")
        if self.track_index <= 0:
            raise ValueError("track_index must be positive")
        if self.duration_frames <= 0:
            raise ValueError("duration_frames must be positive")
        if self.stream_count <= 0:
            raise ValueError("stream_count must be positive")
        if not 0 <= self.stream_index < self.stream_count:
            raise ValueError(
                "stream_index must satisfy 0 <= stream_index < stream_count"
            )
        if self.channels < 0:
            raise ValueError("channels must be non-negative")


@dataclass(frozen=True)
class AudioSlice:
    source_id: str
    start_frame: int
    end_frame: int
    source_in_frame: int
    source_out_frame: int
    segment_id: str | None = None

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError("timeline slice must have a positive range")
        if self.source_in_frame < 0 or self.source_out_frame <= self.source_in_frame:
            raise ValueError("source slice must have a positive range")
        if (
            self.end_frame - self.start_frame
            != self.source_out_frame - self.source_in_frame
        ):
            raise ValueError("timeline and source slice durations must match")
