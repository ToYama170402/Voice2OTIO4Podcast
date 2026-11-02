"""End-to-end orchestration for transcription and XML generation."""

from collections.abc import Iterable

from .evaluator import evaluate_segment
from .fcp_xml import write_xmeml
from .models import ProcessedSegment, RawSegment, seconds_to_frame
from .segmentator import segment_raw


def process_segments(
    raw_segments: Iterable[RawSegment],
    audio_path: str,
    output_path: str,
    fps: int = 30,
    sequence_name: str = "Podcast_Timeline",
) -> list[ProcessedSegment]:
    """Convert raw transcription segments and write their FCP7 XML."""
    processed: list[ProcessedSegment] = []
    previous_end: float | None = None
    for raw_segment in raw_segments:
        for refined in segment_raw(raw_segment):
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
                )
            )
            previous_end = refined.end
    if not processed:
        raise ValueError("transcription produced no segments")
    write_xmeml(output_path, processed, audio_path, sequence_name, fps)
    return processed
