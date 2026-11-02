"""Frame-accurate continuous audio slices for OTIO output."""

from collections.abc import Iterable

from .models import AudioSlice, ProcessedSegment


def build_audio_slices(
    segments: Iterable[ProcessedSegment],
    source_id: str,
    duration_frames: int | None = None,
) -> list[AudioSlice]:
    """Cover a source timeline continuously, including gaps between captions."""
    ordered = sorted(
        (segment for segment in segments if segment.source_id == source_id),
        key=lambda segment: (segment.start_frame, segment.end_frame),
    )
    if not ordered:
        if duration_frames is None or duration_frames <= 0:
            raise ValueError("duration_frames is required when there are no segments")
        return [AudioSlice(source_id, 0, duration_frames, 0, duration_frames)]

    end_of_source = duration_frames or max(segment.end_frame for segment in ordered)
    if end_of_source < ordered[-1].end_frame:
        raise ValueError("duration_frames cannot be shorter than a segment")

    slices: list[AudioSlice] = []
    cursor = 0
    for segment in ordered:
        if segment.start_frame < cursor:
            raise ValueError("segments overlap after frame conversion")
        if segment.start_frame > cursor:
            slices.append(
                AudioSlice(
                    source_id, cursor, segment.start_frame, cursor, segment.start_frame
                )
            )
        slices.append(
            AudioSlice(
                source_id,
                segment.start_frame,
                segment.end_frame,
                segment.in_frame,
                segment.out_frame,
                segment.segment_id,
            )
        )
        cursor = segment.end_frame
    if cursor < end_of_source:
        slices.append(
            AudioSlice(source_id, cursor, end_of_source, cursor, end_of_source)
        )
    return slices
