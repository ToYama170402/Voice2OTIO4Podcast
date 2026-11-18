"""End-to-end orchestration for transcription and timeline generation."""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .evaluator import Evaluation, evaluate_segment
from .filler_detector import FillerRegion, token_filler_regions
from .models import (
    AudioSource,
    ClipMarker,
    ProcessedSegment,
    RawSegment,
    RefinedSegment,
    seconds_to_frame,
    seconds_to_frame_range,
)
from .otio import write_otio
from .restart_detector import RestartRegion, token_restart_regions
from .segmentator import segment_source
from .srt import write_srt

_FILLER_MARKER_NAME = "フィラー候補"
_MIN_MARKER_SECONDS = 0.08
_RESTART_EDGE_TOLERANCE_S = 0.2
_TIME_EPSILON_S = 0.05


def _clip_markers(
    regions: Sequence[FillerRegion],
    start_frame: int,
    end_frame: int,
    fps: int,
) -> list[ClipMarker]:
    """Clip-relative markers for filler candidates overlapping a segment."""
    markers: list[ClipMarker] = []
    segment_start = start_frame / fps
    segment_end = end_frame / fps
    for region in regions:
        overlap_start = max(region.start, segment_start)
        overlap_end = min(region.end, segment_end)
        if overlap_end - overlap_start < _MIN_MARKER_SECONDS:
            continue
        markers.append(
            ClipMarker(
                name=_FILLER_MARKER_NAME,
                start_frame=seconds_to_frame(overlap_start, fps) - start_frame,
                end_frame=seconds_to_frame(overlap_end, fps) - start_frame,
            )
        )
    return markers


def _restart_pieces(
    refined: RefinedSegment,
    restart_regions: Sequence[RestartRegion],
    evaluation: Evaluation,
) -> list[tuple[float, float, str, Evaluation]]:
    """Split a refined segment around restart attempt regions.

    Returns (start, end, text, evaluation) pieces that tile the
    segment. A region covering at least half of the segment flags the
    whole segment. A region lying strictly inside splits the segment
    so only the abandoned attempt becomes a cut candidate; the split
    is skipped when the segment is already a candidate for another
    reason or has no word timestamps to take the piece texts from.
    """
    whole = [(refined.start, refined.end, refined.text, evaluation)]
    overlapping = [
        region
        for region in restart_regions
        if region.end > refined.start and region.start < refined.end
    ]
    if not overlapping:
        return whole
    duration = refined.end - refined.start
    for region in overlapping:
        covered = min(region.end, refined.end) - max(region.start, refined.start)
        if covered >= 0.5 * duration:
            flagged = Evaluation(True, (*evaluation.reasons, region.reason))
            return [(refined.start, refined.end, refined.text, flagged)]
    if evaluation.is_cut_candidate:
        best = max(
            overlapping,
            key=lambda region: (
                min(region.end, refined.end) - max(region.start, refined.start)
            ),
        )
        merged = Evaluation(True, (*evaluation.reasons, best.reason))
        return [(refined.start, refined.end, refined.text, merged)]
    if not refined.words:
        return whole
    inside = [
        region
        for region in overlapping
        if not (
            region.start <= refined.start + _TIME_EPSILON_S
            and region.end >= refined.end - _TIME_EPSILON_S
        )
    ]
    if not inside:
        return whole
    region = inside[0]
    attempt_start = (
        refined.start
        if region.start - refined.start < _RESTART_EDGE_TOLERANCE_S
        else region.start
    )
    attempt_end = (
        refined.end
        if refined.end - region.end < _RESTART_EDGE_TOLERANCE_S
        else region.end
    )

    def piece_text(start: float, end: float) -> str:
        return "".join(
            word.word
            for word in refined.words
            if start - 1e-9 <= word.start < end - 1e-9
        )

    pieces: list[tuple[float, float, str, Evaluation]] = []
    if attempt_start > refined.start:
        pieces.append(
            (
                refined.start,
                attempt_start,
                piece_text(refined.start, attempt_start),
                Evaluation(False, ()),
            )
        )
    pieces.append(
        (
            attempt_start,
            attempt_end,
            piece_text(attempt_start, attempt_end),
            Evaluation(True, (region.reason,)),
        )
    )
    if attempt_end < refined.end:
        pieces.append(
            (
                attempt_end,
                refined.end,
                piece_text(attempt_end, refined.end),
                Evaluation(False, ()),
            )
        )
    return pieces


def process_segments(
    raw_segments: Iterable[RawSegment],
    audio_path: str,
    output_path: str,
    fps: int = 30,
    sequence_name: str = "Podcast_Timeline",
    srt_output_path: str | None = None,
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
        srt_output_path,
    )


def process_sources(
    sources: Iterable[tuple[AudioSource, Iterable[RawSegment]]],
    output_path: str,
    fps: int = 30,
    sequence_name: str = "Podcast_Timeline",
    srt_output_path: str | None = None,
    filler_regions: Mapping[str, Sequence[FillerRegion]] | None = None,
) -> list[ProcessedSegment]:
    """Process multiple one-file-per-speaker sources into one OTIO timeline."""
    processed: list[ProcessedSegment] = []
    source_list = list(sources)
    for source, raw_segments in source_list:
        raw_list = list(raw_segments)
        regions = (filler_regions or {}).get(source.source_id, ())
        if filler_regions is not None:
            # Enrich the acoustic candidates with the token-level scan
            # (Sudachi filler words in the word timestamps). Token
            # regions that overlap an acoustic candidate are dropped so
            # a single utterance does not produce two markers.
            token_regions = token_filler_regions(raw_list)
            regions = list(regions) + [
                region
                for region in token_regions
                if not any(
                    min(region.end, other.end) - max(region.start, other.start)
                    > _TIME_EPSILON_S
                    for other in regions
                )
            ]
        refined_list = list(segment_source(raw_list, track_index=source.track_index))
        restart_regions = token_restart_regions(raw_list)
        previous_end: float | None = None
        for refined in refined_list:
            evaluation = evaluate_segment(refined, previous_end=previous_end)
            pieces = _restart_pieces(refined, restart_regions, evaluation)
            for piece_index, (p_start, p_end, p_text, p_evaluation) in enumerate(
                pieces
            ):
                frame_range = seconds_to_frame_range(
                    p_start,
                    p_end,
                    fps,
                    source.duration_frames,
                )
                if frame_range is None:
                    continue
                start_frame, end_frame = frame_range
                processed.append(
                    ProcessedSegment(
                        segment_id=(
                            refined.segment_id
                            if piece_index == 0
                            else f"{refined.segment_id}~{piece_index}"
                        ),
                        track_index=refined.track_index,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        in_frame=start_frame,
                        out_frame=end_frame,
                        subtitle_text=p_text,
                        is_cut_candidate=p_evaluation.is_cut_candidate,
                        flag_label=p_evaluation.flag_label,
                        marker_reason=p_evaluation.marker_reason,
                        color_label="Red" if p_evaluation.is_cut_candidate else "None",
                        source_id=source.source_id,
                        audio_path=source.path,
                        clip_markers=_clip_markers(
                            regions, start_frame, end_frame, fps
                        ),
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
    write_srt(
        srt_output_path or str(Path(output_path).with_suffix(".srt")),
        processed,
        fps,
    )
    return processed
