"""Core data models shared by the processing stages."""

from dataclasses import dataclass, field
import math


def seconds_to_frame(seconds: float, fps: int) -> int:
    """Convert seconds to the nearest timeline frame."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be a finite non-negative number")
    if fps <= 0:
        raise ValueError("fps must be positive")
    return round(seconds * fps)


@dataclass(frozen=True)
class WordTimestamp:
    word: str
    start: float
    end: float
    probability: float

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
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
        if self.start < 0 or self.end <= self.start:
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
        if self.start < 0 or self.end <= self.start:
            raise ValueError("segment must satisfy 0 <= start < end")
        if self.track_index <= 0:
            raise ValueError("track_index must be positive")


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
class AudioSource:
    source_id: str
    path: str
    track_index: int
    duration_frames: int
    name: str = ""

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if not self.path:
            raise ValueError("path must not be empty")
        if self.track_index <= 0:
            raise ValueError("track_index must be positive")
        if self.duration_frames <= 0:
            raise ValueError("duration_frames must be positive")


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
