"""SubRip subtitle generation."""

from collections.abc import Iterable

from .models import ProcessedSegment


def _timestamp(frame: int, fps: int) -> str:
    total_milliseconds = round(frame * 1000 / fps)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def write_srt(
    output_path: str,
    segments: Iterable[ProcessedSegment],
    fps: int = 30,
) -> None:
    """Write processed subtitle segments as a UTF-8 SubRip file."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    ordered_segments = sorted(
        segments,
        key=lambda segment: (
            segment.start_frame,
            segment.track_index,
            segment.end_frame,
        ),
    )
    with open(output_path, "w", encoding="utf-8", newline="\n") as output:
        for index, segment in enumerate(ordered_segments, start=1):
            output.write(f"{index}\n")
            output.write(
                f"{_timestamp(segment.start_frame, fps)} --> "
                f"{_timestamp(segment.end_frame, fps)}\n"
            )
            output.write(f"{segment.subtitle_text}\n\n")
