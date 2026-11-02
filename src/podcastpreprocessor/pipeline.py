"""End-to-end orchestration for transcription and XML generation."""

from collections.abc import Iterable

from .evaluator import evaluate_segment
from .models import AudioSource, ProcessedSegment, RawSegment, seconds_to_frame
from .otio import write_otio
from .segmentator import segment_raw


def process_segments(
    raw_segments: Iterable[RawSegment],
    audio_path: str,
    output_path: str,
    fps: int = 30,
    sequence_name: str = "Podcast_Timeline",
) -> list[ProcessedSegment]:
    """Convert one source and write an OTIO timeline."""
    raw_list = list(raw_segments)
    duration_frames = max(
        (seconds_to_frame(segment.end, fps) for segment in raw_list),
        default=0,
    )
    return process_sources(
        [(AudioSource("source_1", audio_path, 1, max(duration_frames, 1)), raw_list)],
        output_path,
        fps,
        sequence_name,
    )


def process_sources(
    sources: Iterable[tuple[AudioSource, Iterable[RawSegment]]],
    output_path: str,
    fps: int = 30,
    sequence_name: str = "Podcast_Timeline",
) -> list[ProcessedSegment]:
    """Process multiple one-file-per-speaker sources into one OTIO timeline."""
    processed: list[ProcessedSegment] = []
    source_list = list(sources)
    for source, raw_segments in source_list:
        previous_end: float | None = None
        for raw_segment in raw_segments:
            for refined in segment_raw(raw_segment, track_index=source.track_index):
                evaluation = evaluate_segment(refined, previous_end=previous_end)
                processed.append(
                    ProcessedSegment(
                        segment_id=refined.segment_id,
                        track_index=refined.track_index,
                        start_frame=seconds_to_frame(refined.start, fps),
                        end_frame=seconds_to_frame(refined.end, fps),
                        in_frame=seconds_to_frame(refined.start, fps),
                        out_frame=seconds_to_frame(refined.end, fps),
                        subtitle_text=refined.text,
                        is_cut_candidate=evaluation.is_cut_candidate,
                        flag_label=evaluation.flag_label,
                        marker_reason=evaluation.marker_reason,
                        color_label="Red" if evaluation.is_cut_candidate else "None",
                        source_id=source.source_id,
                        audio_path=source.path,
                    )
                )
                previous_end = refined.end
    if not processed:
        raise ValueError("transcription produced no segments")
    write_otio(
        output_path,
        processed,
        [source for source, _ in source_list],
        sequence_name,
        fps,
    )
    return processed
